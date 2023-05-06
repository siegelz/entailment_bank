import matplotlib.pyplot as plt
import os
import json
import numpy as np

# Loads data from subdirectories of output_dir. Each subdirectory contains a file called scores-test.metrics.json
# which contains the scores for each model on the test set. The subdirectory name is used as the label for the model.
# Returns a dictionary mapping model names to a list of scores.
def load_data(output_dir):
    data = {}
    for subdir in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, subdir)):
            with open(os.path.join(output_dir, subdir, "scores-test.metrics.json")) as f:
                data[subdir] = json.load(f)
    return data

# Loads the scores from a particular model
def load_scores(data, model_name):
    return data[model_name]['outputs/' + model_name + '/scores-test']['QAHC->P']

# Prints the scores for a particular model in table format
# | Leaves F1 | Leaves Accuracy | Steps F1 | Steps Accuracy | Intermediate F1 | Intermediate Accuracy | Overall |
def print_scores(scores):
    print("| {:^10} | {:^15} | {:^10} | {:^15} | {:^15} | {:^22} | {:^15} |".format(
        "Leaves F1", "Leaves Accuracy", "Steps F1", "Steps Accuracy", "Intermediate F1", 
        "Intermediate Accuracy", "Overall"))

    print("| {:^10.4f} | {:^15.4f} | {:^10.4f} | {:^15.4f} | {:^15.4f} | {:^22.4f} | {:^15.4f} |".format(
        scores['proof-leaves']['F1'], scores['proof-leaves']['acc'],
        scores['proof-steps']['F1'], scores['proof-steps']['acc'],
        scores['proof-intermediates']['BLEURT_F1'], scores['proof-intermediates']['BLEURT_acc'],
        scores['proof-overall']['acc']))

# Plots the scores across different models
# Inputs: 
#   Scores - a list of dictionaries mapping model names to scores, each entry of dict is result from different model
#   Metrics - a list of metrics to plot (which correspond to the keys in the dictionary)
# Outputs:
#   A line plot of the scores for each model. The x-axis is the model name, and the y-axis is the score.
def plot_scores(model_names, scores, metrics):
    fig, ax = plt.subplots()
    x = np.arange(len(model_names))
    width = 0.2
    for i in range(len(metrics)):
        metric_name = 'acc' if metrics[i] != 'proof-intermediates' else 'BLEURT_acc'
        ax.bar(x + i * width, [scores[j][metrics[i]][metric_name] for j in range(len(model_names))], width, label=metrics[i])

    ax.set_xlabel('# of Train Examples', size=22)
    ax.set_ylabel('Accuracy', size=22)
    # ax.set_title('Few Shot Performance of T5 (Greedy) + GPT (Search)', size=16)
    ax.set_xticks(x)
    ax.set_xticklabels([model_name.split("_")[1] for model_name in model_names])
    ax.legend(loc='lower right', prop={'size': 16})
    plt.savefig('scores.pdf')
    plt.show()

# Print the scores for stepwise_test, gpt_test, and gpt_stepwise
data = load_data('outputs')
for model_name in ['stepwise_test', 'gpt_test', 'gpt_stepwise', 'real_gpt']:
    print(model_name)
    scores = load_scores(data, model_name)
    print_scores(scores)
print()


# Plot the scores for gpt_2, gpt_4, gpt_6, and gpt_8
model_names = ['gpt_2', 'gpt_4', 'gpt_6', 'gpt_8']
model_metrics = ['proof-leaves', 'proof-steps', 'proof-intermediates', 'proof-overall'] # MISSING PROOF-INTERMEDIATE
scores = [load_scores(data, model_name) for model_name in model_names]

print(scores)
plot_scores(model_names, scores, model_metrics)
