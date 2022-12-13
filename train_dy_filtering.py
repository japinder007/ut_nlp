"""
Reads the file produced by run_with_train_dynamics.py (train_stats.py).  Analyzes the outputs of
train_stats.csv and sorts the examples by descending variability. The indices of top-K
training examples with highest variability are written to a file. This file can be passed to run.py
using the --train_indices_file option. run.py will then train in the usual way but only using the
examples whose indices are in the passed file.
"""
import argparse
import pandas as pd
import os
import csv
import logging
import tqdm
import torch
from collections import defaultdict
import numpy as np
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def read_training_dynamics(file_name: os.path, num_epochs=6):
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
        - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    train_dynamics = {}
    with open(file_name, "r") as f:
        train_dynamics_csv = csv.reader(f, delimiter=",")
        for row_num, row in enumerate(train_dynamics_csv):
            # Skip the header
            if row_num == 0:
                continue
            
            epoch, example_index, logit1, logit2, logit3, label = row
            epoch = int(epoch)
            label = int(label)
            example_index = int(example_index)
            logit1, logit2, logit3 = float(logit1), float(logit2), float(logit3)
            if example_index not in train_dynamics:
                train_dynamics[example_index] = {}
                train_dynamics[example_index]['gold'] = label
                train_dynamics[example_index]['logits'] = [[] for _ in range(num_epochs)]
            
            assert epoch < num_epochs, f"epoch out of range {epoch} {num_epochs}"
            assert label == train_dynamics[example_index]['gold'], f"found contradictory labels for {example_index} {label} and {train_dynamics[example_index]['gold']}"
            train_dynamics[example_index]['logits'][epoch] = [logit1, logit2, logit3]
        
    return train_dynamics

def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)

def compute_forgetfulness(correctness_trend: List[float]) -> int:
  """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
  if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
      return 1000
  learnt = False  # Predicted correctly in the current epoch.
  times_forgotten = 0
  for is_correct in correctness_trend:
    if (not learnt and not is_correct) or (learnt and is_correct):
      # nothing changed.
      continue
    elif learnt and not is_correct:
      # Forgot after learning at some point!
      learnt = False
      times_forgotten += 1
    elif not learnt and is_correct:
      # Learnt!
      learnt = True
  return times_forgotten

def compute_train_dy_metrics(training_dynamics, args):
  """
  Given the training dynamics (logits for each training instance across epochs), compute metrics
  based on it, for data map coorodinates.
  Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
  the last two being baselines from prior work
  (Example Forgetting: https://arxiv.org/abs/1812.05159 and
   Active Bias: https://arxiv.org/abs/1704.07433 respectively).
  Returns:
  - DataFrame with these metrics.
  - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
  """
  confidence_ = {}
  variability_ = {}
  threshold_closeness_ = {}
  correctness_ = {}
  forgetfulness_ = {}

  # Functions to be applied to the data.
  variability_func = lambda conf: np.std(conf)
  if args.include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
    variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
  threshold_closeness_func = lambda conf: conf * (1 - conf)

  loss = torch.nn.CrossEntropyLoss()

  num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])
  if args.burn_out < num_tot_epochs:
    logger.info(f"Computing training dynamics. Burning out at {args.burn_out} of {num_tot_epochs}. ")
  else:
    logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
  logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

  logits = {i: [] for i in range(num_tot_epochs)}
  targets = {i: [] for i in range(num_tot_epochs)}
  training_accuracy = defaultdict(float)

  for guid in tqdm.tqdm(training_dynamics):
    correctness_trend = []
    true_probs_trend = []

    record = training_dynamics[guid]
    # Some training examples might not have data for all epochs, skip these.
    if not all(record['logits']):
        continue

    for i, epoch_logits in enumerate(record["logits"]):
      probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
      true_class_prob = float(probs[record["gold"]])
      true_probs_trend.append(true_class_prob)

      prediction = np.argmax(epoch_logits)
      is_correct = (prediction == record["gold"]).item()
      correctness_trend.append(is_correct)

      training_accuracy[i] += is_correct
      logits[i].append(epoch_logits)
      targets[i].append(record["gold"])

    if args.burn_out < num_tot_epochs:
      correctness_trend = correctness_trend[:args.burn_out]
      true_probs_trend = true_probs_trend[:args.burn_out]

    correctness_[guid] = compute_correctness(correctness_trend)
    confidence_[guid] = np.mean(true_probs_trend)
    variability_[guid] = variability_func(true_probs_trend)

    forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
    threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

  # Should not affect ranking, so ignoring.
  epsilon_var = np.mean(list(variability_.values()))

  column_names = ['guid',
                  'index',
                  'threshold_closeness',
                  'confidence',
                  'variability',
                  'correctness',
                  'forgetfulness',]
  df = pd.DataFrame([[guid,
                      i,
                      threshold_closeness_[guid],
                      confidence_[guid],
                      variability_[guid],
                      correctness_[guid],
                      forgetfulness_[guid],
                      ] for i, guid in enumerate(correctness_)], columns=column_names)

  df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                            training_accuracy[i] / len(training_dynamics)
                            ] for i in range(num_tot_epochs)],
                          columns=['epoch', 'loss', 'train_acc'])
  return df, df_train


def plot_data_map(dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    logger.info(f"Plotting figure for {title} using the {model} model ...")

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        plot.set_title(f"{title}-{model} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{plot_dir}/{title}_{model}.pdf' if show_hist else f'figures/compact_{title}_{model}.pdf'
    fig.savefig(filename, dpi=300)
    logger.info(f"Plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",
                        action="store_true",
                        help="Whether to filter data subsets based on specified `metric`.")
    parser.add_argument("--plot",
                        action="store_true",
                        help="Whether to plot data maps and save as `pdf`.")
    parser.add_argument("--plots_dir",
                        default="./cartography/",
                        type=os.path.abspath,
                        help="Directory where plots are to be saved.")
    parser.add_argument('--metric',
                        choices=('threshold_closeness',
                                'confidence',
                                'variability',
                                'correctness',
                                'forgetfulness'),
                        help="Metric to filter data by.",)
    parser.add_argument("--include_ci",
                        action="store_true",
                        help="Compute the confidence interval for variability.")
    parser.add_argument("--filtering_output_dir",
                        "-f",
                        default="./filtered/",
                        type=os.path.abspath,
                        help="Output directory where filtered datasets are to be written.")
    parser.add_argument("--worst",
                        action="store_true",
                        help="Select from the opposite end of the spectrum acc. to metric,"
                            "for baselines")
    parser.add_argument("--both_ends",
                        action="store_true",
                        help="Select from both ends of the spectrum acc. to metric,")
    parser.add_argument("--burn_out",
                        type=int,
                        default=6,
                        help="# Epochs for which to compute train dynamics.")

    args = parser.parse_args()
    training_dynamics = read_training_dynamics('/home/jpsingh/nlp/fp-dataset-artifacts/train_stats.csv')
    train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)
    # train_dy_filename = os.path.join(args.model_dir, f"td_metrics{burn_out_str}.jsonl")
    train_dy_filename = 'train_dy_metrics.json'
    train_dy_metrics.to_json(train_dy_filename,
                           orient='records',
                           lines=True)
    # plot_data_map(train_dy_metrics, args.plots_dir, title='snli', show_hist=True, model='electra')
    # import pdb; pdb.set_trace()
    sorted_scores = train_dy_metrics.sort_values(by=["variability"], ascending=False)
    num_samples = len(sorted_scores) // 3
    selected = sorted_scores.head(n=num_samples+1)
    selection_iterator = tqdm.tqdm(range(len(selected)))
    with open(f"top_variability_33_ids.csv", "w") as outfile:
      for idx in selection_iterator:
        selected_id = int(selected.iloc[idx]["guid"])
        outfile.write(f"{selected_id}\n")

# import pdb; pdb.set_trace()

