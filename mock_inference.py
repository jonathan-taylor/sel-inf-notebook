import copy
import json
import nbformat
import numpy as np
import uuid

import preprocessors

# Read a notebook (on which to test the preprocessor)
nbpath = 'notebooks/hello-world-dataframe-r.ipynb'
#nbpath = 'notebooks/hello-world-dataframe.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

# Initialize the analysis preprocessor (for full dataset)
analysis_pp = preprocessors.AnalysisPreprocessor(timeout=600)
resources = {}  # empty dict to store outputs etc.

# Analysis -------------------------------------------------------------

# Numpy array to store selection and sufficient stat

# Preprocess the notebook; save info into `resources`
nb, resources = analysis_pp.preprocess(nb, resources=resources)

# Selection variable
selection_type = resources['selection_type']
if selection_type == 'set':
    selection_var = 'set_selection'
elif selection_type == 'fixed':
    selection_var = 'fixed_selection'
else:
    print('WARNING: Unspecified selection type')
print("Selection Type:", selection_type)

# Save selection and sufficient statistic
selected_vars_init = resources[selection_var]['selected_vars']
suff_stat_init = resources['suff_stat']

print("Suff Stat:\n", suff_stat_init, "\n")

#print(selected_vars_init)
#print(suff_stat_init)

print("\n-- ANALYSIS COMPLETE --\n")

# Simulation -----------------------------------------------------------

n_simulations = 3

# Generate empty numpy arrays to fill with simulated selection and
# suff stat
selected_vars_dim = np.shape(selected_vars_init)
selected_vars_sim = np.empty((n_simulations,) + selected_vars_dim)

suff_stat_dim = np.shape(suff_stat_init)
suff_stat_sim = np.empty((n_simulations,) + suff_stat_dim)

# Initialize the simulation preprocessor (for simulated data)
simulate_pp = preprocessors.SimulatePreprocessor(timeout=600)
simulate_pp.data_name = analysis_pp.data_name
print("Pre-simulation")
nb, resources = simulate_pp.preprocess(nb, resources=resources,
                                       km=analysis_pp.km)

for i in range(n_simulations):
    # Preprocess and save results
    nb, resources = simulate_pp.preprocess(nb, resources=resources,
                                           km=simulate_pp.km)
    selected_vars = resources[selection_var]['selected_vars']
    suff_stat = resources['suff_stat']

    selected_vars_sim[i] = selected_vars
    suff_stat_sim[i] = suff_stat
    print("Suff Stat:\n", suff_stat, "\n")
    print("\n-- SIMULATION %s COMPLETE --\n" % (i + 1))

# Cleaning up / shutting down ------------------------------------------

# Shut down the kernel
# NOTE: We only need to apply these commands to `simulate_pp` and not
# `analysis_pp` because the kernel manager from `analysis_pp` gets
# passed to `simulate_pp`.
simulate_pp.kc.stop_channels()
simulate_pp.km.shutdown_kernel(now=simulate_pp.shutdown_kernel == 'immediate')

#for attr in ['nb', 'km', 'kc']:
#    delattr(simulate_pp, attr)

# NOTE: Some issues arise when running the script more than once.
# It seems that this script does not kill the R sessions it creates -
# we need to run `killall R` in bash.

# Selection indicators -------------------------------------------------

print("Selection Type:", selection_type)

# 1D array of indicators for selection event for each simulation
selection = np.empty(n_simulations)

# 'Lee'-type selection - selected
if selection_type == 'set':
    pass

# 'Liu'-type selection - full
elif selection_type == 'fixed':
    for i in range(n_simulations):
        selection[i] = np.array_equal(selected_vars_sim[i], selected_vars_init)

else:
    print('WARNING: Unspecified selection type')

print(selection)
