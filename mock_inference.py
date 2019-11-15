import copy
import json
import nbformat
import numpy as np
import uuid

import preprocessors

# Read a notebook (on which to test the preprocessor)
#nbpath = 'notebooks/hello-world-dataframe.ipynb'
nbpath = 'notebooks/hello-world-dataframe-r.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

# Initialize the analysis preprocessor (for full dataset)
analysis_pp = preprocessors.AnalysisPreprocessor(timeout=600, nb_log_name='analysis_log.ipynb')
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

n_simulations = 5

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
                                                 nb_log_name='simulate_log.ipynb')
print(simulate_pp.analysis_data_name, 'blah')
print(simulate_pp.data_name, 'blah2')
#simulate_pp.data_name = analysis_pp.data_name
print("Pre-simulation")
nb, resources = simulate_pp.preprocess(nb, 
                                       resources=resources,
                                       km=analysis_pp.km)
print('indicators')
print(resources['indicators'])
# so we can run the log notebook
simulate_pp.nb_log.cells = (analysis_pp.nb_log.cells + 
                            simulate_pp.nb_log.cells)
nbformat.write(simulate_pp.nb_log, open('simulate_log_final.ipynb', 'w'))
print('first pass done')
for i in range(n_simulations):
    # Preprocess and save results
    nb, resources = simulate_pp.preprocess(nb, 
                                           resources=resources,
                                           km=simulate_pp.km)
    indicators = resources['indicators']
    suff_stat = resources['suff_stat']

    print("Suff Stat:\n", suff_stat, "\n")
    print("\n-- SIMULATION %s COMPLETE --\n" % (i + 1))

# original_selection = resources['original_selection']

# # Cleaning up / shutting down ------------------------------------------

# # Shut down the kernel
# # NOTE: We only need to apply these commands to `simulate_pp` and not
# # `analysis_pp` because the kernel manager from `analysis_pp` gets
# # passed to `simulate_pp`.
# simulate_pp.kc.stop_channels()
# simulate_pp.km.shutdown_kernel(now=simulate_pp.shutdown_kernel == 'immediate')

# #for attr in ['nb', 'km', 'kc']:
# #    delattr(simulate_pp, attr)

# # NOTE: Some issues arise when running the script more than once.
# # It seems that this script does not kill the R sessions it creates -
# # we need to run `killall R` in bash.

# # Selection indicators -------------------------------------------------

# print("Selection Type:", selection_type)

# def get_fixed_sel_indicator(original_sel_vars, simulated_sel_vars):
#     return np.array_equal(simulated_sel_vars, original_sel_vars)


# def get_set_sel_indicator(original_sel_vars, simulated_sel_vars):
#     n_vars = original_sel_vars.shape[0]
#     indicators = np.zeros(n_vars)
#     # TODO: fix this part, which actually generates positive indicators
#     """
#     for i in range(n_vars):
#         #indicators[i] = np.isin(original_sel_vars[i], simulated_sel_vars)
#         for j in range(simulated_sel_vars.shape[0]):
#             if(simulated_sel_vars[j] == original_sel_vars[i]):
#                 indicators[i] = 1
#     """
#     return indicators


# if selection_type == 'full':
#     indicators = np.empty(n_simulations)
#     for i in range(n_simulations):
#         indicators[i] = \
#                 get_fixed_sel_indicator(original_selection, selected_vars_sim[i])


# elif selection_type == 'set':
#     n_vars = original_selection.shape[0]
#     indicators = np.empty((n_simulations, n_vars))
#     for i in range(n_simulations):
#         indicators[i] = \
#                 get_set_sel_indicator(original_selection, selected_vars_sim[i])

# print(indicators)

# """
# # 'Lee'-type selection - selected
# if selection_type == 'set':
#     # TODO: Right now this is just a copy of full selection (below)
#     for i in range(n_simulations):
#         selection[i] = np.array_equal(selected_vars_sim[i], selected_vars_init)

# # 'Liu'-type selection - full
# elif selection_type == 'fixed':
#     for i in range(n_simulations):
#         selection[i] = np.array_equal(selected_vars_sim[i], selected_vars_init)
# """
