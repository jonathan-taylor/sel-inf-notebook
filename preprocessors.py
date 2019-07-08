import copy, uuid, json
import numpy as np

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets import Unicode, Int, default

class SelectiveInferencePreprocessor(ExecutePreprocessor):
    """
    Notebook preprocessor for selective inference.
    """

    pass

    data_name = Unicode()
    default_index = Int(-1)

    @default('data_name')
    def _default_data_name(self):
        return 'data_' + str(uuid.uuid1()).replace('-','') 

    define_data = Unicode()
    @default('define_data')
    def _default_define_data(self):
        if self.km.kernel_name == 'python3':
            source = '\n'.join(['if "%(data)s" not in locals():', '    %(data)s = {}']) % {'data':self.data_name}
        elif self.km.kernel_name == 'ir':
            source = 'if (! exists("%(data)s") { %(data)s = list() }' % {'data':self.data_name}
        return source

    def capture_selection(self, cell, resources):
        """
        Capture JSON-serialized selection outputs
        in resources['set_selection'] and resources['fixed_selection']
        """
        set_selection, fixed_selection = resources['set_selection'], resources['fixed_selection']

        if 'capture_selection' in cell.metadata:
            capture_cell = nbformat.v4.new_code_cell()
            for selection in cell.metadata['capture_selection']:
                capture_cell.source += 'print(%s)\n' % selection['name']
            selection_outputs = self.run_cell(capture_cell, self.default_index)
            for selection, output in zip(cell.metadata['capture_selection'],
                                         selection_outputs):
                {'set':set_selection,
                 'fixed':fixed_selection}[selection['selection_type']].setdefault(selection['name'], json.loads(output['text']))

class AnalysisPreprocessor(SelectiveInferencePreprocessor):

    """
    This preprocessor runs the analysis on the
    collected data, capturing output that will
    be conditioned on.
    """

    force_raise_errors = True
    cell_allows_errors = False

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
        resources.setdefault('data_name', self.data_name)

        # Original code from execute.py
        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources

        # capturing the data

        cell = self.prepend_data_input_code(cell)
        outputs = self.run_cell(cell, cell_index)
        
        # capturing selection

        self.capture_selection(cell, resources)

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

        return cell, resources

    def datafile_input_source(self, variable_name, data_file):
        if self.km.kernel_name == 'python3':
            source = '''
%(variable_name)s = np.loadtxt("%(data_file)s", delimiter=',')
%(data_name)s["%(variable_name)s"] = %(variable_name)s
''' % {'data_name':self.data_name, 'data_file':data_file, 'variable_name':variable_name}
        elif self.km.kernel_name == 'ir':
            source = '''
%(variable_name)s = read.table("%(data_file)s", delimiter=',', header=TRUE)
%(data_name)s["%(variable_name)s"] = %(variable_name)s
''' % {'data_name':self.data_name, 'data_file':data_file, 'variable_name':variable_name}
        return source

    def prepend_data_input_code(self, cell):
        """
        Create a new cell, prepending code to read
        in data identified in cell metadata.

        If no data input, return cell without modification.
        """
        if 'data_input' in cell.metadata:
            cell_cp = copy.copy(cell)
            cell_cp.source = self.define_data + '\n'
            for variable_name, data_file in cell.metadata['data_input']:
                cell_cp.source += self.datafile_input_source(variable_name, data_file) + '\n'
            cell_cp.source += cell.source                        
            return cell_cp
        else:
            return cell

class SimulatePreprocessor(SelectiveInferencePreprocessor):
    """
    This preprocessor reruns the analysis
    several times, capturing the observed
    values of variables conditioned on in
    the analysis.
    """

    simulate_data = Unicode()

    def simulate_data(self, resources):
        # all of the collected data will have to be available at runtime if we want to bootstrap, say
        if self.km.kernel_name == 'python3':
            source = '\n'.join(['if "%(simulate_data)s" not in locals():', '    %(simulate_data)s = %(simulate)s(%(data_name)s, "%(fixed_selection)s")']) 
        elif self.km.kernel_name == 'ir':
            source = 'if (! exists("%(data)s") { %(data)s = list() }' % {'data':self.data_name}

        source = source % {'simulate_data':self.simulate_data,
                           'simulate':resources.get('data_model', {}).get('resample_data', 'function_not_found'),
                           'data_name':self.data_name,
                           'fixed_selection': json.dumps(resources.get("fixed_selection", {}))}
        simulate_cell = nbformat.v4.new_code_cell(source=source)
        self.run_cell(simulate_cell, self.default_index)

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
        
        self.simulate_data(resources)
        resources.setdefault('fixed_selection', {})
        resources.setdefault('set_selection', {})
        resources.setdefault('data_model', {})
        resources.setdefault('data_name', self.data_name)

        # Original code from execute.py
        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources

        # capturing the data

        cell = self.prepend_simulation_data_code(cell)
        outputs = self.run_cell(cell, cell_index)
        
        # capturing selection

        self.capture_selection(cell, resources)

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

        return cell, resources

    def prepend_data_input_code(self, cell):
        """
        Create a new cell, prepending code to read
        in data identified in cell metadata.

        If no data input, return cell without modification.
        """
        if 'data_input' in cell.metadata:
            cell_cp = copy.copy(cell)
            cell_cp.source = self.define_data + '\n'
            for variable_name, data_file in cell.metadata['data_input']:
                cell_cp.source += self.datafile_input_source(variable_name, data_file) + '\n'
            cell_cp.source += cell.source                        
            return cell_cp
        else:
            return cell

    def capture_sufficient_statistics(self, resources):
        if self.km.kernel_name == 'python3':
            source = '''
%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, """%(fixed_selection)s""");
print(','.join([str(s) for s in %(suff_stat)s]))
''' 
        elif self.km.kernel_name == 'ir':
            source = '''
%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, "%(fixed_selection)s");
print(cat(%(suff_stat)s, sep=","))
'''
        capture_cell = nbformat.v4.new_code_cell()
        if 'data_model' in resources and 'sufficient_statistics' in resources['data_model']:
            capture_cell.source = source % {'suff_stat': 'suff_stat_' + str(uuid.uuid1()).replace('-',''),
                                            'data_name': self.data_name, 
                                            'fixed_selection': json.dumps(resources['fixed_selection']),
                                            'suff_stat_map': resources['data_model']['sufficient_statistics'] 
                                            }
            outputs = self.run_cell(capture_cell, 0)
            sufficient_stat = np.array([float(s) for s in outputs[0]['text'].strip().split(',')])
            print(capture_cell.source)
            return sufficient_stat
        else:
            return np.array([])
    



