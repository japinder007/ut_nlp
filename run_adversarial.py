"""
This file is used to generate universal trigger tokens.
Usage:
  python3 run_adversarial.py --output_dir ./trained_model_filtered
"""

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
from attack import hotflip_attack
import torch
from torch.nn.modules.sparse import Embedding
from typing import List, Tuple
import heapq
from operator import itemgetter
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

NUM_PREPROCESSING_WORKERS = 2

def add_trigger_tokens_to_batch(batch, trigger_token_ids: List[int]):
  # batch is assumed to have the following fields - input_ids', token_type_ids, attention_mask, labels
  # batch['input_ids'].shape, batch['token_type_ids'].shape,  batch['attention_mask'].shape
  # All shapes are B, S where B is the batch length an S is the max sequence length.
  if not trigger_token_ids:
    # No change is needed.
    return batch
  
  new_batch = {}
  device = batch['input_ids'].device
  # Make sure all required fields are present in the batch.
  assert all(f in batch for f in ['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
  batch_size, seq_len = batch['input_ids'].shape
  new_batch['labels'] = batch['labels']
  num_trigger_tokens = len(trigger_token_ids)
  trigger_token_tensor = torch.tensor(trigger_token_ids, dtype=batch['input_ids'].dtype).repeat(batch_size, 1)
  trigger_token_tensor = trigger_token_tensor.to(device)
  assert trigger_token_tensor.shape == (batch_size, num_trigger_tokens), "Unexpected shape of trigger token tensor"
  new_batch['input_ids'] = torch.cat((trigger_token_tensor, batch['input_ids']), dim=1)
  assert new_batch['input_ids'].shape == (batch_size, num_trigger_tokens + seq_len)
  new_batch['token_type_ids'] = torch.cat((torch.zeros(batch_size, num_trigger_tokens, dtype=batch['token_type_ids'].dtype).to(device), batch['token_type_ids']), dim=1)
  assert new_batch['token_type_ids'].shape == (batch_size, num_trigger_tokens + seq_len)
  new_batch['attention_mask'] = torch.cat((torch.ones(batch_size, num_trigger_tokens, dtype=batch['attention_mask'].dtype).to(device), batch['attention_mask']), dim=1)
  assert new_batch['attention_mask'].shape == (batch_size, num_trigger_tokens + seq_len)
  return new_batch


def get_loss_per_candidate(
  trainer, model, batch, token_index: int, 
  trigger_token_ids : List[int], candidate_trigger_token_ids: List[List[int]]
) -> List[Tuple[List[int], float]]:

  loss_per_candidate = []
  
  new_batch = add_trigger_tokens_to_batch(batch, trigger_token_ids)  
  loss = trainer.compute_loss(model, new_batch)
  loss_per_candidate.append((trigger_token_ids, loss.item()))

  for c_index in candidate_trigger_token_ids[token_index]:
    new_batch['input_ids'][:,token_index] = c_index
    loss = trainer.compute_loss(model, new_batch)
    # print(f"Loss {i}: {loss}")
    new_tokens = deepcopy(trigger_token_ids)
    new_tokens[token_index] = c_index
    loss_per_candidate.append((new_tokens, loss.item()))
  
  return loss_per_candidate


def beam_search(trainer, model, batch, trigger_token_ids: List[int], candidate_trigger_token_ids, beam_size: int):
  # Perform beam search over the best candidates.
  # candidate_trigger_token_ids has shape T, num_candidates.
  assert len(candidate_trigger_token_ids.shape) == 2, "Expecting (T, n_cand) shape"
  assert len(trigger_token_ids) == candidate_trigger_token_ids.shape[0]

  loss_per_candidate = get_loss_per_candidate(trainer, model, batch, 0, trigger_token_ids, candidate_trigger_token_ids)
  top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
  for i in range(1, len(trigger_token_ids)):
    loss_per_candidate = []
    for cand, _ in top_candidates:
      loss_per_candidate.extend(get_loss_per_candidate(trainer, model, batch, i, cand, candidate_trigger_token_ids))
    top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
  return min(top_candidates, key=itemgetter(1))[0]


# Use this method to create a new evaluation dataset from the existing one by prefixing with trigger_token_ids.
# def add_trigger_tokens(example, trigger_token_ids_str: str):
def add_trigger_tokens(examples, trigger_token_ids):  
  input_ids = []
  token_type_ids = []
  attention_masks = []
  for ids, tts, ams in zip(examples['input_ids'], examples['token_type_ids'], examples['attention_mask']):
    input_ids.append(trigger_token_ids + ids)
    token_type_ids.append([0] * len(trigger_token_ids) + tts)
    attention_masks.append([1] * len(trigger_token_ids) + ams)

  examples['input_ids'] = input_ids
  examples['token_type_ids'] = token_type_ids
  examples['attention_mask'] = attention_masks
  return examples


def add_trigger_tokens_premise(examples):  
  # trigger_token_ids = [int(t) for t in trigger_token_ids_str.split()]
  premises = []
  for premise in examples["premise"]:
    premises.append('hello ' + premise)

  examples['premise'] = premises
  return examples


def change_label(examples, label):  
  examples['label'] = [label] * len(examples['label'])
  return examples

def print_premise_hypothesis(dataset: datasets.arrow_dataset.Dataset):
  for premise, hypothesis in zip(dataset['premise'], dataset['hypothesis']):
    print(f"Premise: {premise}, hypothesis: {hypothesis}")

def print_mismatches(eval_dataset: datasets.arrow_dataset.Dataset, predicted_labels_orig: np.ndarray, predicted_labels_with_trigger: np.ndarray, mismatching_indices: np.ndarray, orig_label:int, trigger_label: int, num_examples: int):
  indices = np.where((predicted_labels_orig[mismatching_indices] == orig_label) & (predicted_labels_with_trigger[mismatching_indices] == trigger_label))[0]
  print(f"\nPrinting examples: orig label {orig_label} -> trigger label {trigger_label} {len(indices)}")
  print_premise_hypothesis(eval_dataset[indices[:num_examples]])


def compare_eval_group_performance(eval_dataset: datasets.arrow_dataset.Dataset, trainer: Trainer, eval_dataset_featurized: datasets.arrow_dataset.Dataset, trigger_token_ids: List[int], num_examples: int) -> Tuple[np.ndarray, np.ndarray]:
  # Compare eval group performance with and without trigger and return a list of ids where the
  # two differ. The ids can be used to get the actual examples.
  logits, labels, _ = trainer.predict(eval_dataset_featurized)
  assert len(logits.shape) == 2, "Expecting logits to have shape N x C"

  predicted_labels_orig = np.argmax(logits, axis=1)
  eval_dataset_with_trigger_tokens_featurized = eval_dataset_featurized.map(
    lambda examples: add_trigger_tokens(examples, trigger_token_ids),
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS
  )
  logits_with_trigger, labels_with_trigger, _ = trainer.predict(eval_dataset_with_trigger_tokens_featurized)
  assert len(logits_with_trigger.shape) == 2, "Expecting logits to have shape N x C"
  predicted_labels_with_trigger = np.argmax(logits_with_trigger, axis=1)

  matching_indices = np.asarray(predicted_labels_orig == predicted_labels_with_trigger).nonzero()[0]
  mismatching_indices = np.asarray(predicted_labels_orig != predicted_labels_with_trigger).nonzero()[0]

  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 0, 1, num_examples)
  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 0, 2, num_examples)
  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 1, 0, num_examples)
  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 1, 2, num_examples)
  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 2, 0, num_examples)
  print_mismatches(eval_dataset, predicted_labels_orig, predicted_labels_with_trigger, mismatching_indices, 2, 1, num_examples)
  return matching_indices, mismatching_indices
  

def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--compare_eval_set_performance', action='store_true',
                      help='If true, print out examples from the evaluation set where the model with and without triggers differ.')
    argp.add_argument('--trigger_tokens', type=str, default=None,
                      help="comma separated list of trigger tokens. (no spaces). Example: 'hello,world'")
    argp.add_argument('--num_examples_to_print', type=int, default=10,
                      help="Number of examples to print. Only used when compare_eval_set_performance is True'")
    

    training_args, args = argp.parse_args_into_dataclasses()

    model_class = AutoModelForSequenceClassification
    model_path = './trained_model'
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    task_kwargs = {'num_labels': 3}
    model = model_class.from_pretrained(model_path, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    compute_metrics = compute_accuracy    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)
    
    max_length = 128
    prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, max_length)
    dataset_id = ('snli',)
    dataset = datasets.load_dataset(*dataset_id)
    # remove SNLI examples with no label
    dataset = dataset.filter(lambda ex: ex['label'] != -1)

    eval_split = 'validation'
    eval_dataset = dataset[eval_split]
    print(f"Initial evaluation dataset has {len(eval_dataset)} examples")
    # For the eval dataset only keep the entailment examples so that we can compare how 
    # that class is impacted by the trigger.
    eval_dataset = eval_dataset.filter(lambda ex: ex['label'] == 0)
    print(f"evaluation dataset has {len(eval_dataset)} entailment examples")
    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )
    if args.compare_eval_set_performance:
      assert args.trigger_tokens, "Trigger tokens should be specified"
      trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
      )
      matching_indices, mismatching_indices = compare_eval_group_performance(eval_dataset, trainer, eval_dataset_featurized, [tokenizer.vocab[t] for t in args.trigger_tokens.split(",")], args.num_examples_to_print) 
      matching_examples = eval_dataset[matching_indices[:args.num_examples_to_print]]
      # mismatching_examples = eval_dataset[mismatching_indices[:args.num_examples_to_print]]
      print(f"\nMatching examples {args.num_examples_to_print} of {len(matching_indices)}")
      print_premise_hypothesis(matching_examples)
      # print(f"\nMis matching examples {args.num_examples_to_print} of {len(mismatching_indices)}")
      # print_premise_hypothesis(mismatching_examples)
      return
    
    max_length = 128
    train_dataset = dataset['train']

    # Filter out the entailment examples. We will label them as contradiction to generate the
    # trigger tokens.
    print(f"Number of original data points {len(train_dataset)}")
    train_dataset = train_dataset.filter(lambda ex: ex['label'] == 0)
    print(f"Number of entailment data points  {len(train_dataset)}")

    # Map the label of the entailment data points to contradiction.
    train_dataset = train_dataset.map(
      lambda examples: change_label(examples, 2),
      batched=True,
      num_proc=NUM_PREPROCESSING_WORKERS
    )
    print(f"Changed labels from entailment to contradiction")
    
    # Tokenize the datasets.
    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names
    ) 
    print(f"Tokenized the dataset")

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    
    # Register hooks to collect the gradient of the embedding module.
    def get_vocab_embedding_module(model):
        for m in model.modules():
            if isinstance(m, Embedding) and m.num_embeddings == 30522:
                return m
        return None

    embedding_module = get_vocab_embedding_module(model)
    extracted_grads = []
    def extract_grad_hook(module, grad_in, grad_out):
        extracted_grads.append(grad_out[0])
    embedding_module.register_backward_hook(extract_grad_hook)

    train_data_loader = trainer.get_train_dataloader()


    # results = trainer.evaluate(**eval_kwargs)
    # print('Evaluation results:')
    # print(results)

    # Start the attack. (https://github.com/Eric-Wallace/universal-triggers/blob/master/snli/snli.py)
    # 1. Add hooks to collect the gradient for the embeddings.
    # 2. Filter the training set to only contain examples from the entailment class.
    # 3. For the examples selected in (1), set the label to contradiction
    # 4. Get the original accuracy 
    # 5. Switch the model to training mode.
    # 6. Decide on the number of triggers and initialize triggers.
    # 7. for each batch (we might need to pass a hook to the trainer or use the batches as is)
    # 7a.   set model to train mode.
    # 7b.   get gradient of triggers.
    # 7c.   find attack candidates.
    # ...

    # Add hooks to collect the gradient of the embeddings.
    trigger_tokens = ['the', 'the', 'the']
    # trigger_tokens = ['the', 'the']
    # trigger_tokens = ['the']
    trigger_token_ids = [tokenizer.vocab[t] for t in trigger_tokens]
    num_candidates = 5
    beam_size = 5

    embedding_matrix = embedding_module.weight.cpu().detach()
    eval_batch_frequency = 100

    results = trainer.evaluate()
    print(f'Initial evaluation accuracy: {results}')

    for batch_idx, batch in enumerate(train_data_loader):
      # Move the tensors to gpu.
      for k in batch.keys():
        batch[k] = batch[k].to(model.device)

      print(f"Starting batch {batch_idx} {tokenizer.convert_ids_to_tokens(trigger_token_ids)}")
      model.train()
      model.zero_grad()
      extracted_grads = []
      loss = trainer.compute_loss(model, batch)
      loss.backward()

      # batch has shape (B, S, D) where B=batch size, S=sequence length, D=embedding dimension.
      # average_grad has shape (S, D) after taking the average.
      average_grad = torch.sum(extracted_grads[0].cpu(), dim=0) / batch['input_ids'].shape[0]
      # average grad will have shape (T, D) where T is the number of trigger tokens.
      average_grad = average_grad[:len(trigger_token_ids)]
      average_grad = average_grad.unsqueeze(0)

      gradient_dot_embedding = torch.einsum("bij,kj->bik", (average_grad, embedding_matrix))
      gradient_dot_embedding *= -1 
      # best_k_ids has shape (B, T, num_candidates).
      # For each (B, T) pair, it finds the top num_candidates. Thus, for each token
      # it is independently finding the maximum.
      _, best_k_ids = torch.topk(gradient_dot_embedding, num_candidates, dim=2)
      # candidate_trigger_token_ids has shape (T, num_candidates)
      candidate_trigger_token_ids = best_k_ids.detach().cpu().numpy()[0]

      trigger_token_ids = beam_search(trainer, model, batch, trigger_token_ids, candidate_trigger_token_ids, beam_size)
      if (batch_idx + 1) % eval_batch_frequency == 0:
        eval_dataset_with_trigger_tokens_featurized = eval_dataset_featurized.map(
        lambda examples: add_trigger_tokens(examples, trigger_token_ids),
          batched=True,
          num_proc=NUM_PREPROCESSING_WORKERS
        )
        results = trainer.evaluate(eval_dataset=eval_dataset_with_trigger_tokens_featurized)
        print(f"Accuracy after batch {batch_idx}: {results}")



    eval_dataset_with_trigger_tokens_featurized = eval_dataset_featurized.map(
      lambda examples: add_trigger_tokens(examples, trigger_token_ids),
      batched=True,
      num_proc=NUM_PREPROCESSING_WORKERS
    )
    results = trainer.evaluate(eval_dataset=eval_dataset_with_trigger_tokens_featurized)
    print(f"Accuracy after batch {batch_idx}, {tokenizer.convert_ids_to_tokens(trigger_token_ids)}: {results}")
        


if __name__ == "__main__":
  main()

