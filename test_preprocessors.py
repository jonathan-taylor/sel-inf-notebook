from preprocessors import *
import copy, uuid, json
import numpy as np

import nbformat


def check_data_model(resources):

    # this has to be a cell
    data_star = resources['data_model']['resample_data'](resources['data'],
                                                         resources['fixed_selection'])
    suff_stat_star = resources['data_model']['sufficient_statistics'](resources['data'],
                                                                      resources['fixed_selection'])
    targets_star = resources['data_model']['estimators'](resources['data'],
                                                         resources['fixed_selection'],
                                                         resources['set_selection'])

# ## Custom Preprocessor Tests
# 
# https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/base.py
# 
# According to the docstring for `preprocess()` in `nbconvert.preprocessors.base`,
# 
#  > If you wish to apply your preprocessing to each cell, you might want to override preprocess_cell method instead.
# 
# Therefore, we focus on writing a custom `preprocess_cell()` function in our subclass.

# In[14]:


# Read the drop the losers notebook
#nbpath = 'notebooks/hello-world-r.ipynb'
nbpath = 'notebooks/hello-world-dataframe-r.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

mypp = AnalysisPreprocessor(timeout=600)
resources = {}
mypp.preprocess(nb, resources = resources)
print(resources)
resources['set_selection'] = {'selected_vars': [int(i) for i in resources['set_selection']['selected_vars']['selection']]}

source = """
print(%(data)s)
%(resample)s(%(data)s, '%(fixed)s')
%(sufficient)s(%(data)s, '%(fixed)s')
%(target)s(%(data)s, '%(fixed)s', '%(set)s')
""" % {'data': resources['data_name'],
       'resample': resources['data_model']['resample_data'],
       'sufficient': resources['data_model']['sufficient_statistics'],
       'target': resources['data_model']['estimators'],
       'fixed': json.dumps(resources['fixed_selection']),
       'set': json.dumps(resources['set_selection'])}
test_cell = nbformat.v4.new_code_cell(source=source)
nb.cells.append(test_cell)
print(mypp.preprocess_cell(test_cell, resources, 0), 'aha')


