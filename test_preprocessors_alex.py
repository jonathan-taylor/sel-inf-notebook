import copy
import json
import nbformat
import numpy as np
import uuid

import preprocessors

# Read a notebook (on which to test the preprocessor)
nbpath = 'notebooks/hello-world-dataframe-r.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

# Initialize a preprocessor for selective inference
mypp = preprocessors.AnalysisPreprocessor(timeout=600)
resources = {}  # empty dict to store outputs etc.

# Preprocess the notebook; save info into `resources`
mypp.preprocess(nb, resources = resources)
print(resources)

"""
sim_pp = preprocessors.SimulatePreprocessor(timeout=600)
resources = {}
sim_pp.preprocess(nb, resources = resources)
print(resources)
"""
