"""
This file implements the dataset cartography paper. It trains the model for a number of epochs.
During each epoch, for each training example, it tracks the logits of all the classes.
The logits for each training epoch are written to a file (train_stats.csv).

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
import pandas as pd
from tqdm.auto import tqdm

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

    training_args, args = argp.parse_args_into_dataclasses()

    model_class = AutoModelForSequenceClassification
    model_path = './trained_model'
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    task_kwargs = {'num_labels': 3}
    model = model_class.from_pretrained(args.model, **task_kwargs)
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
    import pdb; pdb.set_trace()
    dataset = dataset.filter(lambda ex: ex['label'] != -1)

    NUM_PREPROCESSING_WORKERS = 2
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

    max_length = 128
    train_dataset = dataset['train']
    
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
    
    eval_batch_frequency = 100

    results = trainer.evaluate()
    print(f'Initial evaluation accuracy: {results}')

    batch_size = 128
    example_indices = np.arange(len(train_dataset_featurized))
    epochs = 6
    num_batches = len(train_dataset_featurized) // batch_size

    num_training_steps = epochs * num_batches
    trainer.create_optimizer_and_scheduler(num_training_steps)
    # Create a dataframe with columns: epoch, example_index, logit1, logit2, logit3, label
    df = pd.DataFrame(columns=["epoch", "example_index", "logit1", "logit2", "logit3", "label"])

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(epochs):
        np.random.shuffle(example_indices)
        for batch_idx in range(num_batches):
            start_index = batch_idx * batch_size
            current_indices = example_indices[start_index:start_index+batch_size]
            batch = train_dataset_featurized[current_indices]

            # Convert lists to tensors and move them to the device.
            for k in batch.keys():
                batch[k] = torch.tensor(batch[k], device=model.device)
            batch['labels'] = batch['label']
            batch.pop('label')

            model.train()
            model.zero_grad()
            loss = trainer.compute_loss(model, batch)
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, batch: {batch_idx}, train loss: {loss}")
            loss.backward()
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad()
            progress_bar.update(1)

            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            labels = batch['labels'].detach().cpu().numpy()

            batch_df = pd.DataFrame({
                "epoch": [epoch] * batch_size,
                "example_index": current_indices,
                "logit1": logits[:,0],
                "logit2": logits[:,1],
                "logit3": logits[:,2],
                "label": labels
            })
            batch_df = batch_df.astype({'epoch': 'int32', 'example_index': 'int32', 'label': 'int32'})
            df = pd.concat([df, batch_df], ignore_index=True)

            if batch_idx % eval_batch_frequency == 0:
                results = trainer.evaluate()
                print(f"Accuracy after batch {batch_idx}: {results}")

    # Write the data frame to a file.
    df = df.astype({'epoch': 'int32', 'example_index': 'int32', 'label': 'int32'})
    df.to_csv(training_args.output_dir + f'/train_stats.csv', index=False)        


if __name__ == "__main__":
  main()
