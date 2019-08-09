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

# Preprocess the notebook; save info into `resources`
nb, resources = analysis_pp.preprocess(nb, resources=resources)
print("-- FINAL ANALYSIS RESOURCES --\n", resources)
print("\n-- ANALYSIS COMPLETE --\n")

# Initialize the simulation preprocessor (for simulated data)
simulate_pp = preprocessors.SimulatePreprocessor(timeout=600)
simulate_pp.data_name = analysis_pp.data_name
nb, resources = simulate_pp.preprocess(nb, resources=resources,
                                       km=analysis_pp.km)
print("-- FINAL SIMULATION RESOURCES --\n", resources)
print("\n-- SIMULATION COMPLETE --\n")

# Shut down the kernel
# NOTE: We only need to apply these commands to `simulate_pp` and not
# `analysis_pp` because the kernel manager from `analysis_pp` gets
# passed to `simulate_pp`.
simulate_pp.kc.stop_channels()
simulate_pp.km.shutdown_kernel(now=simulate_pp.shutdown_kernel == 'immediate')

#for attr in ['nb', 'km', 'kc']:
#    delattr(simulate_pp, attr)
