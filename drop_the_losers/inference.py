import copy, os
import json
import nbformat
import numpy as np, pandas as pd
import uuid
import pickle
import preprocessors
from selectinf.learning.fitters import gbm_fit_sk
from selectinf.learning.core import keras_fit

# simulate some data
df1 = pd.DataFrame({'X1':np.random.standard_normal(100), 
                    'X2':np.random.standard_normal(100), 
                    'X3':np.random.standard_normal(100)})
df1.to_csv('/Users/jonathantaylor/git-repos/sel-inf-notebook/drop_the_losers/stage1.csv', index=False)
winner = ['X1', 'X2', 'X3'][np.argmax(np.asarray(df1.mean(0)))]
df2 = pd.DataFrame({winner:np.random.standard_normal(80)})
df1.to_csv('/Users/jonathantaylor/git-repos/sel-inf-notebook/drop_the_losers/stage2.csv', index=False)

# Read a notebook (on which to test the preprocessor)
#nbpath = 'notebooks/hello-world-dataframe.ipynb'
nbpath = 'drop_the_losers/drop-the-losers.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

# Initialize the analysis preprocessor (for full dataset)
analysis_pp = preprocessors.AnalysisPreprocessor(timeout=600, nb_log_name='analysis_log_drop.ipynb')
resources = {}  # empty dict to store outputs etc.

# Analysis -------------------------------------------------------------

# Numpy array to store selection and sufficient stat

# Preprocess the notebook; save info into `resources`
nb, resources = analysis_pp.preprocess(nb, resources=resources)
print(resources)
print('estimates')
print(resources['estimates'])

# # Selection variable
# selection_type = resources['selection_type']
# if selection_type == 'set':
#     selection_var = 'set_selection'
# elif selection_type == 'fixed':
#     selection_var = 'fixed_selection'
# else:
#     print('WARNING: Unspecified selection type')
# print("Selection Type:", selection_type)

# Save selection and sufficient statistic
# selected_vars_init = resources[selection_var]['selected_vars']
suff_stat_init = resources['suff_stat']

print("Suff Stat:\n", suff_stat_init, "\n")

#print(selected_vars_init)
#print(suff_stat_init)

print("\n-- ANALYSIS COMPLETE --\n")

# Simulation -----------------------------------------------------------

n_simulations = 2000

# Generate empty numpy arrays to fill with simulated selection and
# suff stat
# selected_vars_dim = np.shape(selected_vars_init)
# selected_vars_sim = np.empty((n_simulations,) + selected_vars_dim,
#                              dtype=object)

# suff_stat_dim = np.shape(suff_stat_init)
# suff_stat_sim = np.empty((n_simulations,) + suff_stat_dim,
#                          dtype=object)

# Initialize the simulation preprocessor (for simulated data)
simulate_pp = preprocessors.SimulatePreprocessor(timeout=600,
                                                 analysis_selection_list_name=analysis_pp.selection_list_name,
                                                 analysis_data_name=analysis_pp.data_name,
                                                 nb_log_name='drop_the_losers/simulate_log_drop.ipynb')

observed_suff_stat = analysis_pp._capture(analysis_pp.sufficient_stat_name)

#simulate_pp.data_name = analysis_pp.data_name
print("Pre-simulation")
nb, resources = simulate_pp.preprocess(nb, 
                                       resources=resources,
                                       km=analysis_pp.km)
final_source = ''
for cell in simulate_pp.nb_log.cells:
    final_source += '\n#BEGINCELL\n\n'
    indented_cell = ['    ' + l for l in cell.source.split('\n')]
    final_source += '\n'.join(indented_cell)
    final_source += '\n#ENDCELL\n\n'

fn_source = 'data_analysis_scooby = function(%s) {\n %s ; return(%s) }\n' % (simulate_pp.collector, final_source, simulate_pp.collector)
simulate_pp.run_cell(nbformat.v4.new_code_cell(source=fn_source), 0)
simulate_pp.run_cell(nbformat.v4.new_code_cell(source='for(i in 1:%(nsim)d) { %(col)s = data_analysis_scooby(%(col)s)}' % 
                                               {'nsim':n_simulations,
                                                'col':simulate_pp.collector}), 0)

print('indicators')
print(resources['indicators'])
# so we can run the log notebook
simulate_pp.nb_log.cells = (analysis_pp.nb_log.cells + 
                            simulate_pp.nb_log.cells)
nbformat.write(simulate_pp.nb_log, open('drop_the_losers/simulate_log_drop_final.ipynb', 'w'))
print('first pass done')

indicators = simulate_pp._capture('%s[["%s"]]' % (simulate_pp.collector, simulate_pp.indicator_name))
suff_stats = simulate_pp._capture('%s[["%s"]]' % (simulate_pp.collector, simulate_pp.sufficient_stat_name))
resources['sim_indicators'] = indicators
resources['sim_suff_stats'] = suff_stats
resources['observed_suff_stat'] = observed_suff_stat
pickle.dump(resources, open('drop_the_losers/drop_info.pckl', 'wb'))

value = preprocessors.inference(resources, keras_fit, 
                                fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})
new_df = pd.DataFrame({'pivot':[value[0][0]]})
import pandas as pd

if os.path.exists('results.csv'):
    df = pd.read_csv('results.csv')
    df = pd.concat([df, new_df])
else:
    df = new_df
df.to_csv('results.csv', index=False)
