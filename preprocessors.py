import copy
import json
import nbformat
import numpy as np
import uuid
from contextlib import contextmanager
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets import Unicode, Int, Bool, default
from json_dataframe import base64_to_dataframe

#from json_dataframe import dataframe_to_json, json_to_dataframe, dataframe_to_jsonR, base64_to_dataframe

class SelectiveInferencePreprocessor(ExecutePreprocessor):
    """Notebook preprocessor for selective inference. Executes cells and
    stores certain outputs based on cell metadata.
    """
    @contextmanager
    def setup_preprocessor(self, nb, resources, km=None):
        """Context manager for setting up the class to execute a
        notebook.

        This assigns the input `nb` to `self.nb` where it will be
        modified in-place. It also creates and assigns the Kernel
        Manager (`self.km`) and Kernel Client(`self.kc`).

        It is intended to yield to a block that will execute codeself.

        When control returns from the yield it stops the client's zmq
        channels, shuts down the kernel, and removes the now unused
        attributes.

        Parameters
        ----------
        nb : NotebookNode
            Notebook being executed.
        resources : dictionary
            Additional resources used in the conversion process. For
            example, passing ``{'metadata': {'path': run_path}}`` sets
            the execution path to ``run_path``.
        km : KernerlManager (optional)
            Optional kernel manaher. If none is provided, a kernel
            manager will be created.

        Returns
        -------
        nb : NotebookNode
            The executed notebook.
        resources : dictionary
            Additional resources used in the conversion process.
        """
        path = resources.get('metadata', {}).get('path', '') or None
        self.nb = nb

        # clear display_id map
        self._display_id_map = {}
        self.widget_state = {}
        self.widget_buffers = {}

        if km is None:
            self.km, self.kc = self.start_new_kernel(cwd=path)
            try:
                # Yielding unbound args for more easier understanding
                # and downstream consumption
                yield nb, self.km, self.kc
            finally:
                # Below code commented out so that we don't stop the kernel

                #self.kc.stop_channels()
                #self.km.shutdown_kernel(now=self.shutdown_kernel == 'immediate')

                #for attr in ['nb', 'km', 'kc']:
                #    delattr(self, attr)
                print("hereiam?"*10)
                pass
        else:
            self.km = km
            if not km.has_kernel:
                km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
            self.kc = km.client()

            self.kc.start_channels()
            try:
                self.kc.wait_for_ready(timeout=self.startup_timeout)
            except RuntimeError:
                self.kc.stop_channels()
                raise
            self.kc.allow_stdin = False
            try:
                yield nb, self.km, self.kc
            finally:
                for attr in ['nb', 'km', 'kc']:
                    delattr(self, attr)


    data_name = Unicode()
    default_index = Int(-1)


    @default('data_name')
    def _default_data_name(self):
        return 'data_' + str(uuid.uuid1()).replace('-','') 


    define_data = Unicode()
    @default('define_data')
    def _default_define_data(self):
        """Generate source code to define data.
        """
        if self.km.kernel_name == 'python3':
            source = ['if "%(data)s" not in locals():', '    %(data)s = {}']
            source = '\n'.join(source) % {'data':self.data_name}
        elif self.km.kernel_name == 'ir':
            source = 'if (! exists("%(data)s")) { %(data)s = list() }'
            source = source % {'data':self.data_name}
        return source


    def capture_selection(self, cell, resources):
        """Capture JSON-serialized selection outputs into
        resources['set_selection'] and resources['fixed_selection'].
        """
        set_selection = resources['set_selection']
        fixed_selection = resources['fixed_selection']

        # Generate a new cell after the cell whose metadata contains
        # the attribute 'capture_selection'
        if 'capture_selection' in cell.metadata:
            capture_cell = nbformat.v4.new_code_cell()
            # TODO: for loop necessary?
            for selection in cell.metadata['capture_selection']:
                if self.km.kernel_name == 'python3':
                    # JSON encoding
                    if selection['encoder'] == 'json':
                        capture_cell.source += '\n'.join(['from IPython.display import display',
                                                          'import json',
                                                          'display({"application/selective.inference":json.dumps(%s)}, metadata={"encoder":"json"}, raw=True)' % selection['name']])
                    # Base 64 dataframe encoding
                    elif selection['encoder'] == 'dataframe':
                        capture_cell.source += '\n'.join(['from IPython.display import display',
                                                          'import json, tempfile, feather',
                                                          'filename = tempfile.mkstemp()[1]',
                                                          'feather.write_dataframe(%s, filename)' % selection['name'],
                                                          'display({"application/selective.inference":open(filename, "rb").read()}, metadata={"encoder":"dataframe"}, raw=True)'])

                elif self.km.kernel_name == 'ir':
                    if selection['encoder'] == 'json':
                        capture_cell.source += '''
    IRdisplay:::display_raw("application/selective.inference", FALSE, toJSON(%s), NULL, list(encoder="json"))
    ''' % selection['name']
                    elif selection['encoder'] == 'dataframe':
                        capture_cell.source += '''
    library(feather)
    filename = tempfile()
    feather::write_feather(%s, filename)
    A = readBin(file(filename, 'rb'), 'raw', file.size(filename) + 1000)
    IRdisplay:::display_raw("application/selective.inference", TRUE, NULL, filename, list(encoder="dataframe"))
    ''' % selection['name']

            _, selection_outputs = self.run_cell(capture_cell, self.default_index)

            # TODO: Why a for loop?
            for selection, output in zip(cell.metadata['capture_selection'],
                                         selection_outputs):
                output_data = output['data']['application/selective.inference'] # a string
                print('OUTPUT:\n', output_data)
                decoder = {'json':json.loads,
                           'dataframe':base64_to_dataframe,
                           }.get(output.metadata['encoder'], 'json')
                {'set':set_selection,
                 'fixed':fixed_selection}[selection['selection_type']].setdefault(selection['name'], decoder(output_data))
                print('DECODER:\n', decoder(output_data))


class AnalysisPreprocessor(SelectiveInferencePreprocessor):
    """This preprocessor runs the analysis on the collected data,
    capturing output that will be conditioned on.
    """
    force_raise_errors = True
    cell_allows_errors = False

    def preprocess_cell(self, cell, resources, cell_index):
        """Executes a single code cell. Must return modified cell and
        resource dictionary.
        
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

        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources

        # capturing the data

        cell = self.prepend_data_input_code(cell)
        _, outputs = self.run_cell(cell, cell_index)
        
        # capturing selection

        self.capture_selection(cell, resources)

        if 'data_model' in cell.metadata:
            for var in ['estimators', 'sufficient_statistics', 'resample_data']:
                resources['data_model'][var] = cell.metadata['data_model'][var] # hooks used to define target and generative model

        cell.outputs = outputs

        return cell, resources


    def datafile_input_source(self, variable_name, data_file):
        """Generate source code to read the notebook's input data. The
        data data source is specified in the notebook's metadata, and
        should have already been extracted by the preprocessor.

        Parameters
        ----------
        variable_name : string
            The name of the variable to which the input data will be
            read.
        data_file : string
            The name of the input data file.

        Returns
        -------
        source : string
            Source code to read the input data file.
        """
        if self.km.kernel_name == 'python3':
            source = [
            '%(variable_name)s = np.loadtxt("%(data_file)s", delimiter=",")',
            '%(data_name)s["%(variable_name)s"] = %(variable_name)s']
            source = '\n'.join(source)
            source = source % {'data_name':self.data_name,
                               'data_file':data_file,
                               'variable_name':variable_name}
        elif self.km.kernel_name == 'ir':
            source = [
            '%(variable_name)s = read.table("%(data_file)s", sep=",", header=TRUE)',
            '%(data_name)s["%(variable_name)s"] = %(variable_name)s']
            source = '\n'.join(source)
            source = source % {'data_name':self.data_name,
                               'data_file':data_file,
                               'variable_name':variable_name}
        return source


    def prepend_data_input_code(self, cell):
        """Create a new cell, prepending code to read in data specified
        in an existing cell's metadata.

        If no data input, return cell without modification.

        Parameters
        ----------
        cell: NotebookNode cell
            The cell for which we will prepend code to read in data.
            This cell should contain the metadata attribute
            'data_input', whose value is _____

        Returns
        -------
        cell_cp: NotebookNode cell
            A copy of the given cell, prepended with code to read in the
            data identified in the given cell's metadata.
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
    """This preprocessor reruns the analysis several times on simulated
    data (using a simulation procedure specified by the user). Each
    time, it captures the observed values of variables on which the
    analysis is conditioned.
    """
    data_has_been_simulated = Bool(False)

    simulated_data = Unicode()
    @default('simulated_data')
    def _default_data_name(self):
        return 'simulated_data_' + str(uuid.uuid1()).replace('-','') 

    def simulate_data(self, resources):
        # all of the collected data will have to be available at runtime if we want to bootstrap, say
        if not self.data_has_been_simulated:
            if self.km.kernel_name == 'python3':
                source = '\n'.join(['if "%(simulated_data)s" not in locals():', '    %(simulated_data)s = %(simulate)s(%(data_name)s, "%(fixed_selection)s")']) 
            elif self.km.kernel_name == 'ir':
                source = 'if (! exists("%(simulate_data)s")) { %(simulated_data)s = %(simulate)s(%(data_name)s, "%(fixed_selection)s") }' % {'data':self.data_name}

            source = source % {'simulated_data':self.simulate_data,
                               'simulate':resources.get('data_model', {}).get('resample_data', 'function_not_found'),
                               'data_name':self.data_name,
                               'fixed_selection': json.dumps(resources.get("fixed_selection", {}))}
            return source
            simulate_cell = nbformat.v4.new_code_cell(source=source)
            self.run_cell(simulate_cell, self.default_index)

            self.data_has_been_simulated = True
        

    def preprocess_cell(self, cell, resources, cell_index):
        """Executes a single code cell. Must return modified cell and
        resource dictionary.
        
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

        # Capturing the data
        cell = self.prepend_simulation_data_code(cell)
        outputs = self.run_cell(cell, cell_index)
        
        # Capture selection
        self.capture_selection(cell, resources)

        # capturing sufficient stats
        self.capture_sufficient_statistics(resources)

        # Populate `resources` dictionary with information from notebook
        # metadata - hooks used to define target and generative model
        if 'data_model' in cell.metadata:
            for var in ['estimators', 'sufficient_statistics', 'resample_data']:
                resources['data_model'][var] = cell.metadata['data_model'][var]

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
        """Create a new cell, prepending code to read in data identified
        in cell metadata.

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
        """Capture sufficient statistic output into
        resources['suff_stat']. Assume suff stat is a data frame.

        Parameters
        ----------
        resources : dictionary
            Information extracted from the notebook's metadata
        """
        # Create a "phantom code cell" to capture the sufficient
        # statistics into a variable in the notebook's session

        # Skeleton for source code to capture sufficient statistics
        if self.km.kernel_name == 'python3':  # source for Python kernel
            source = '''
%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, """%(fixed_selection)s""");
print(','.join([str(s) for s in %(suff_stat)s]))''' 
        elif self.km.kernel_name == 'ir':  # source for R kernel
            source = '''
%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, "%(fixed_selection)s");
print(cat(%(suff_stat)s, sep=","))'''

        # Generate a cell to fill with the source code specified above
        capture_cell = nbformat.v4.new_code_cell()
        if ('data_model' in resources) and \
                ('sufficient_statistics' in resources['data_model']):

            # Apply formatting rules for the cell's source code (i.e.
            # fill w/ variable names)
            suff_stat_var = 'suff_stat_' + str(uuid.uuid1()).replace('-', '')
            capture_cell.source = source % {'suff_stat': suff_stat_var,
                    'data_name': self.data_name, 
                    'fixed_selection': json.dumps(resources['fixed_selection']),
                    'suff_stat_map': resources['data_model']['sufficient_statistics'] 
            }

            # Append additional source code to output the raw base64
            # encoding of the sufficient statistic dataframe (feather)
            if self.km.kernel_name == 'python3':
                capture_cell.source += '\n'.join([
                        'from IPython.display import display',
                        'import json, tempfile, feather',
                        'filename = tempfile.mkstemp()[1]',
                        'feather.write_dataframe(%s, filename)' % suff_stat_var,
                        'display({"application/selective.inference":open(filename, "rb").read()}, metadata={"encoder":"dataframe"}, raw=True)'])
            elif self.km.kernel_name == 'ir':
                source = ['library(feather)',
                          'filename = tempfile()',
                          'feather::write_feather(%s, filename)',
                          'A = readBin(file(filename, "rb"), "raw", file.size(filename) + 1000)',
                          'IRdisplay:::display_raw("application/selective.inference", TRUE, NULL, filename, list(encoder="dataframe"))']
                source = '\n'.join(source) % suff_stat_var
            capture_cell.source += source

            _, suff_stats = self.run_cell(capture_cell, self.default_index)
            print(suff_stats)

            # Base 64 string
            output_data = suff_stats['data']['application/selective.inference']
            print('OUTPUT:\n', output_data)

            # TODO: fix decoder
            decoder = {'json':json.loads,
                       'dataframe':base64_to_dataframe,
                       }.get(output.metadata['encoder'], 'json')
                {'set':set_selection,
                 'fixed':fixed_selection}[selection['selection_type']].setdefault(selection['name'], decoder(output_data))
                print('DECODER:\n', decoder(output_data))

            # Run the cell and capture the sufficient statistic output
            # TODO: turn below into robust capture
            #outputs = self.run_cell(capture_cell, 0)
            #sufficient_stat = np.array([float(s) for s in \
            #        outputs[0]['text'].strip().split(',')])
            print(capture_cell.source)
            return sufficient_stat
        else:
            return np.array([])
            # TODO: throw error message/warning?
