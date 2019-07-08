
# coding: utf-8

# In[3]:


import numpy as np
import nbformat  # read notebooks
from nbconvert.preprocessors import ExecutePreprocessor  # execute notebooks


# # Inference After Drop the Losers Selection
# 
# https://nbconvert.readthedocs.io/en/latest/api/preprocessors.html
# 
# https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/base.py
# 
# This notebooks aims to apply selective inference to `drop-the-losers.ipynb` by parsing notebook cell metadata and using a custom sublass of `nbconvert.preprocessors.ExecutePreprocessor`.

# ## Custom Preprocessor Class Definition
# 
# https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
# 
# Subclassed from `ExecutePreprocessor`

# ### Prototype from meeting

# In[ ]:
import copy, uuid, json

class SelInfPreprocessor(ExecutePreprocessor):
    """Notebook preprocessor for selective inference.
    """
    pass

class SimulatePreprocessor(SelInfPreprocessor):
    pass

class AnalysisPreprocessor(SelInfPreprocessor):

    force_raise_errors = True
    cell_allows_errors = False

    data_name = 'data_' + str(uuid.uuid1()).replace('-','') # should be in __init__

    def preprocess_cell(self, cell, resources, cell_index):
        """Executes a single code cell. Must return modified cell and resource dictionary.
        
        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        index : int
            Index of the cell being processed
        """
        
        resources.setdefault('fixed_selection', {})
        resources.setdefault('set_selection', {})
        resources.setdefault('data_model', {})

        # Original code from execute.py
        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources

        # NEW CODE: updating `resources` based on metadata

        # capturing the data

        cell_cp = copy.copy(cell)

        # python version
        cell_cp.source = '''
if "%(data)s" not in locals():
    %(data)s = {}
''' % {'data': self.data_name}
        if 'data_input' in cell.metadata:
            for name, csvfile in cell.metadata['data_input']:
                cell_cp.source += '''
%(name)s = np.loadtxt("%(csv)s", delimiter=',')
%(data)s["%(name)s"] = %(name)s
''' %{'data': self.data_name, 'name':name, 'csv':csvfile}
        
        cell_cp.source += cell.source

        outputs = self.run_cell(cell_cp, cell_index)  # main info from cell
        
        # capturing selection

        set_selection, fixed_selection = resources['set_selection'], resources['fixed_selection']

        if 'capture_selection' in cell.metadata:
            cell_cp.source = ''
            for selection in cell.metadata['capture_selection']:
                cell_cp.source = 'print(%s)\n' % selection['name']
            selection_outputs = self.run_cell(cell_cp, cell_index)
            for selection, output in zip(cell.metadata['capture_selection'],
                                         selection_outputs):
                {'set':set_selection,
                 'fixed':fixed_selection}[selection['selection_type']].setdefault(selection['name'], json.loads(output['text']))
        
        # finding hooks for data model

        if 'data_model' in cell.metadata:
            for var in ['estimators', 'sufficient_statistics', 'resample_data']:
                resources['data_model'][var] = cell.metadata['data_model'][var] # hooks used to define target and generative model

        cell.outputs = outputs

        if not self.allow_errors:
            for out in outputs:
                if out.output_type == 'error':
                    pattern = u"""\
                        An error occurred while executing the following cell:
                        ------------------
                        {cell.source}
                        ------------------

                        {out.ename}: {out.evalue}
                        """
                    msg = pattern.format(out=out, cell=cell)
                    raise ValueError(msg)

        resources['data_name'] = self.data_name
        return cell, resources


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
nbpath = 'Hello world.ipynb'
nb = nbformat.read(nbpath, nbformat.NO_CONVERT)

mypp = AnalysisPreprocessor(timeout=600, kernel_name='python3')
resources = {}
mypp.preprocess(nb, resources = resources)
print(resources)

source = """
print(%(data)s)
%(resample)s(%(data)s, '''%(fixed)s''')
%(sufficient)s(%(data)s, '''%(fixed)s''')
%(target)s(%(data)s, '''%(fixed)s''', '''%(set)s''')
""" % {'data': resources['data_name'],
       'resample': resources['data_model']['resample_data'],
       'sufficient': resources['data_model']['sufficient_statistics'],
       'target': resources['data_model']['estimators'],
       'fixed': json.dumps(resources['fixed_selection']),
       'set': json.dumps(resources['set_selection'])}
test_cell = nbformat.v4.new_code_cell(source=source)
nb.cells.append(test_cell)
mypp.preprocess(nb, resources)


