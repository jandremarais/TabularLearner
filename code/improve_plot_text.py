from pathlib import Path
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs, display_embs
from matplotlib import pyplot as plt

path = Path('../data/adult')
experiment_name, exp_path = create_experiment('residual_depth', path)

config_df, recorder_df, param_names, metric_names = load_results(exp_path)
summary_df = summarise_results(recorder_df, param_names, metric_names)

# plot_best(summary_df, param_names, metric_names)
# plt.savefig(exp_path/'best2.png', bbox_inches='tight')

plot_over_epochs(summary_df, param_names, metric_names, config_df)
plt.savefig(exp_path/'all_epochs2.png', bbox_inches='tight')