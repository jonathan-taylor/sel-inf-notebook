import copy
import json
import nbformat
import numpy as np
import uuid
from contextlib import contextmanager
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets import Unicode, Int, Bool, default
from json_dataframe import base64_to_dataframe

from jupyter_client.manager import start_new_kernel
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
        km : KernelManager (optional)
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
            kernel_name = nb['metadata']['kernelspec']['name']
            self.km, self.kc = start_new_kernel(cwd=path, 
                                                kernel_name=kernel_name)
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
                pass
        else:
            self.km = km
            if not km.has_kernel:
                km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
            self.kc = km.client()
            #print('has a kernel')
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
                #print("hereiam?"*10)
                pass

    data_name = Unicode()
    selection_list_name = Unicode('NOTCREATED')
    default_index = Int(-1)
    processing_mode = Unicode()
    logstr = Unicode()

    def exec_debug(self, code):
        new_cell = nbformat.v4.new_code_cell()
        new_cell.source = code
        val = self.run_cell(new_cell, 0)
        try:
            print(val[1][0]['data']['text/plain'])
        except:
            print(val[1][0]['text'])

    def run_cell(self, cell, cell_index):
        self.logstr += cell.source
        self.logstr += '\n' + '=' * 40 + '\n'
        return ExecutePreprocessor.run_cell(self, cell, cell_index)

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


    def capture_selection(self, cell, resources, save_to_notebook=False):
        """Capture JSON- or feather/base64-serialized selection outputs
        into resources['set_selection'] and
        resources['fixed_selection'].
        """
        set_selection = resources['set_selection']
        fixed_selection = resources['fixed_selection']

        # Path for injection code
        if self.km.kernel_name == 'python3':
            injection_code_path = 'injection_code/python3-kernel/'
        elif self.km.kernel_name == 'ir':
            injection_code_path = 'injection_code/ir-kernel/'

        # Generate a new cell after the cell whose metadata contains
        # the attribute 'capture_selection'

        if 'capture_selection' in cell.metadata:
            # TODO: reverse the above logic; don't do anything if
            # capture_selection not in cell.metadata
            #print("-- DEBUG: CAPTURE_SELECTION --")
            #print(cell.metadata)

            resources['selection_type'] = cell.metadata['capture_selection'][0]['selection_type']
            capture_cell = nbformat.v4.new_code_cell()

            if self.selection_list_name == 'NOTCREATED':
                self.selection_list_name = 'original_selection_' + str(uuid.uuid1()).replace('-','') 
                if self.km.kernel_name == 'python3':
                    capture_cell.source = ('%s = {};\n' % self.selection_list_name) + capture_cell.source
                elif self.km.kernel_name == 'ir':
                    capture_cell.source = ('%s = list();\n' % self.selection_list_name) + capture_cell.source

            for selection in cell.metadata['capture_selection']:
                if self.km.kernel_name == 'python3':
                    # JSON encoding
                    if selection['encoder'] == 'json':
                        capture_cell.source += '\n'.join(['from IPython.display import display',
                                                          'import json',
                                                          'display({"application/selective.inference":json.dumps(%s)}, metadata={"encoder":"json"}, raw=True)' % selection['name']])
                    # Base 64 dataframe encoding
                    elif selection['encoder'] == 'dataframe':
                        with open(injection_code_path +
                                  'capture_cell_dataframe.txt', 'r') as f:
                            source = f.read()
                        capture_cell.source += source % selection['name']
                    # keep original selection variable persistent
                    capture_cell.source += '%s["%s"] = %s;\n' % (self.selection_list_name,
                                                                 selection['name'],
                                                                 selection['name'])
                elif self.km.kernel_name == 'ir':
                    if selection['encoder'] == 'json':
                        capture_cell.source += 'IRdisplay:::display_raw("application/selective.inference", FALSE, toJSON(%s), NULL, list(encoder="json"));\n' % selection['name']
                    elif selection['encoder'] == 'dataframe':
                        with open(injection_code_path +
                                  'capture_cell_1.txt', 'r') as f:
                            source = f.read()
                        capture_cell.source += source % selection['name']

                    # keep original selection variable persistent
                    capture_cell.source += '%s[["%s"]] = %s;\n' % (self.selection_list_name,
                                                                   selection['name'],
                                                                   selection['name'])

            # Base 64 string of dataframe of selected variables
            _, selection_outputs = self.run_cell(capture_cell, self.default_index)
            #self.exec_debug('print("NAMESPACE")')
            #self.exec_debug('ls()')
            #print('PRINT CELL')
            #print_cell = nbformat.v4.new_code_cell()
            #print_cell.source = 'print(%s)' % self.selection_list_name
            #print('CAPTURE')
            #print(capture_cell.source)
            #print('PRINT')
            #print(print_cell)
            #print(self.run_cell(print_cell, self.default_index))
            #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')
            #print('-'*40)
            # For loop to accomodate for possibly multiple selection events
            for selection, output in zip(cell.metadata['capture_selection'],
                                         selection_outputs):
                # TODO: implement functionality to accomodate for
                # multiple selections

                # Base 64 encoding of dataframe of selected vars
                output_data = output['data']['application/selective.inference'] # a string
                if save_to_notebook:
                    # Save the base 64 encoding to the notebook's metadata
                    cell.metadata['original_selection'] = output_data

                    # Save the actual dataframe to `resources`
                    resources['original_selection'] = base64_to_dataframe(output_data)

                #print('OUTPUT:\n', output_data)
                decoder = {'json':json.loads,
                           'dataframe':base64_to_dataframe,
                           }.get(output.metadata['encoder'], 'json')
                {'set':set_selection,
                 'fixed':fixed_selection}[selection['selection_type']].setdefault(selection['name'], decoder(output_data))
                #print('DECODER:\n', decoder(output_data))


    def capture_sufficient_statistics(self, resources):
        """Capture sufficient statistic output into
        resources['suff_stat']. Assume suff stat is a data frame.

        Parameters
        ----------
        resources : dictionary
            Information extracted from the notebook's metadata.

        Returns
        -------
        resources : dictionary
            An updated resources dictionary.
        """
        # TODO: why isn't `cell` passed in like the other functions?
        # Return an empty array if necessary metadata is missing
        if not (('data_model' in resources) and \
                ('sufficient_statistics' in resources['data_model'])):
            print("-- WARNING --")
            print("Sufficient statistic capture unable to complete.\n")
            return np.array([])

        # Create a "phantom code cell" to capture the sufficient
        # statistics into a variable in the notebook's session

        # Generate an empty code cell
        capture_cell = nbformat.v4.new_code_cell()

        # Variable name (in notebook kernel) for sufficient statistic
        suff_stat_var = 'suff_stat_' + str(uuid.uuid1()).replace('-', '')

        # [Source code for phantom notebook cell]
        # Compute sufficient statistic and save it to a variable
        if self.km.kernel_name == 'python3':  # source for Python kernel
            source = '%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, "%(fixed_selection)s");'
        elif self.km.kernel_name == 'ir':  # source for R kernel
            source = '%(suff_stat)s = %(suff_stat_map)s(%(data_name)s, "%(fixed_selection)s");'
        # Apply formatting rules to the source code (i.e. fill in
        # variable names)

        data_name = self.data_name
        
        capture_cell.source = source % {'suff_stat': suff_stat_var,
                'data_name': data_name, 
                'fixed_selection': json.dumps(resources['fixed_selection']),
                'suff_stat_map': resources['data_model']['sufficient_statistics'] 
        }

        # [Source code for phantom notebook cell]
        # Save the sufficient statistic as a feather file and output
        # the file's raw bytecode as base 64

        # TODO: Add source code to close connections after opening them
        # (opened to make temp files)
        if self.km.kernel_name == 'python3':
            source = '\n'.join([
                    'from IPython.display import display',
                    'import json, tempfile, pandas, feather',
                    'filename = tempfile.mkstemp()[1]',
                    '%s = pandas.DataFrame({"suff_stat":%s})' % (suff_stat_var, suff_stat_var),
                    'feather.write_dataframe(%s, filename)' % suff_stat_var,
                    'display({"application/selective.inference":open(filename, "rb").read()}, metadata={"encoder":"dataframe"}, raw=True)'])
        elif self.km.kernel_name == 'ir':
            source = ['\nlibrary(feather)',
                      'filename = tempfile()',
                      'feather::write_feather(%s, filename)',
                      'f = file(filename, "rb")',
                      'A = readBin(f, "raw", file.size(filename) + 1000)',
                      'IRdisplay:::display_raw("application/selective.inference", TRUE, NULL, filename, list(encoder="dataframe"))',
                      'close(f)',
                      'unlink(filename)']
            source = '\n'.join(source) % suff_stat_var
        capture_cell.source += source

        # Run the phantom cell and save its output (suff stat in base 64)
        _, cell_output = self.run_cell(capture_cell, self.default_index)
        #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')

        # TODO: temp debugging
        #print(cell_output)

        """
        print("\n-- START CHECK --")
        print("SUFF STAT CAPTURE SOURCE:")
        print(capture_cell.source)
        print(cell_output)
        print("-- END CHECK --\n")
        """

        suff_stat_base64 = cell_output[0]['data']['application/selective.inference']
        #print('SUFF STAT BASE 64 OUTPUT:\n', suff_stat_base64)

        # Convert the base 64 output into a dataframe
        suff_stat = base64_to_dataframe(suff_stat_base64)
        #print('SUFF STAT DATAFRAME:\n', suff_stat)

        # Save suff stat dataframe into resources
        resources['suff_stat'] = suff_stat

        return resources


    def recall_original_selection(self, cell, resources):
        """Read the original selected variables from the notebook's
        metadata into the kernel's namespace.
        """
        # TODO: add parsers for python kernel, JSON encoding

        # Currently omitting b/c trying to use `resources` instead
        # TODO: remove this function if `resources` works
        """
        if 'original_selection' not in cell.metadata:
            return
        
        # Base 64 string of dataframe of original selected vars
        original_sel_base64 = cell.metadata['original_selection']

        # Generate a new cell to read the selected vars into the kernel

        # Write the base 64 to disk (as a feather)
        recall_src = ['base64_original_sel <- "%s"',
                'filename <- tempfile()',
                'writeBin("raw")'
                ''] % original_sel_base64
        recall_src = '\n'.join(recall_src)
        recall_cell = nbformat.v4.new_code_cell(source=recall_src)

        # Read the feather from disk
        """


class AnalysisPreprocessor(SelectiveInferencePreprocessor):
    """This preprocessor runs the analysis on the collected data,
    capturing output that will be conditioned on.
    """
    force_raise_errors = True
    cell_allows_errors = False
    processing_mode = Unicode('analysis')

    def preprocess(self, nb, resources=None, km=None):
        """Preprocess notebook executing each code cell. The input
        argument `nb` is modified in-place.

        Parameters
        ----------
        nb : NotebookNode
            Notebook being executed.
        resources : dictionary (optional)
            Additional resources used in the conversion process. For example,
            passing ``{'metadata': {'path': run_path}}`` sets the
            execution path to ``run_path``.
        km: KernelManager (optional)
            Optional kernel manager. If none is provided, a kernel manager will
            be created.

        Returns
        -------
        nb : NotebookNode
            The executed notebook.
        resources : dictionary
            Additional resources used in the conversion process.
        """
        if not resources:
            resources = {}

        with self.setup_preprocessor(nb, resources, km=km):
            self.log.info("Executing notebook with kernel: %s" % self.kernel_name)

            # Preprocess each cell of the notebook
            # TODO: use super(AnalysisPreprocessor)?
            nb, resources = ExecutePreprocessor.preprocess(self, 
                                                           nb, 
                                                           resources,
                                                           km=km)

            # Metadata operations
            # info_msg = self._wait_for_reply(self.kc.kernel_info())
            # nb.metadata['language_info'] = info_msg['content']['language_info']
            # self.set_widgets_metadata()

        # Collect sufficient statistics
        resources = self.capture_sufficient_statistics(resources)

        return nb, resources


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

        """
        print('ANALYSIS CELL SOURCE')
        print('-'*20)
        print(cell.source)
        print('-'*20)
        """
        cell = self.prepend_data_input_code(cell)
        _, outputs = self.run_cell(cell, cell_index)
        #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')

        # Capture selection
        self.capture_selection(cell, resources, True)

        if 'data_model' in cell.metadata and 'functions' in cell.metadata:
            for var in cell.metadata['data_model'].keys():
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

    processing_mode = Unicode('simulation')
    selection_list_name = Unicode('NOTCREATED')
    original_selection_list_name = Unicode('NOTCREATED')

    original_data_name = Unicode()

    @default('data_name')
    def _default_data_name(self):
        return 'simulated_data_' + str(uuid.uuid1()).replace('-','') 

    def simulate_data(self, resources):
        """Simulate data within the client notebook by injecting a cell
        that calls the simulate function.
        """
        #print('simulating data')
        if self.km.kernel_name == 'python3':
            source = '%(simulated_data)s = %(simulate)s(%(data_name)s, "%(fixed_selection)s")' + '\n'
            source += '\n'.join(['for key in %(simulated_data)s.keys():',
                    '    locals()[key] = %(simulated_data)s[key]'])
        elif self.km.kernel_name == 'ir':
            source = '%(simulated_data)s = %(simulate)s(%(data_name)s, "%(fixed_selection)s")'
            source += '\n'.join(['\nfor(key in %(simulated_data)s) {',
                                 '  assign(key, %(simulated_data)s[[key]])',
                                 '}'])

        source = source % {'simulated_data':self.data_name,
                           'simulate':resources.get('data_model', {}).get('resample_data', 'function_not_found'),
                           'data_name':self.original_data_name,
                           'fixed_selection': json.dumps(resources.get("fixed_selection", {}))}
        
        simulate_cell = nbformat.v4.new_code_cell(source=source)
        """
        print('SIMULATE DATA SOURCE simulate_data')
        print('-'*20)
        print(simulate_cell.source)
        print('-'*20)
        """
        self.run_cell(simulate_cell, self.default_index)
        #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')

    def capture_selection_indicators(self, resources):
        """Generate a one-dimensional data frame of selection indicators
        by comparing the simulated selection to the original full-
        sample selection.
        """
        # TODO: complete this function

        # Return an empty array if necessary metadata is missing
        if not (('data_model' in resources) and \
                ('selection_indicator_function' in resources['data_model'])):
            print("-- WARNING --")
            raise ValueError("Unable to generate selection indicators.\n")

        # Generate a new cell after the cell whose metadata contains
        sel_ind_cell = nbformat.v4.new_code_cell()
        result_id = 'result_%s' % str(uuid.uuid1()).replace('-','') 
        sel_ind_cell.source = '%s = %s(%s, %s)\n' % (result_id,
                                                      resources['data_model']['selection_indicator_function'],
                                                      self.original_selection_list_name,
                                                      self.selection_list_name)
        sel_ind_cell.source += 'print(%s)\n' % result_id
        print(sel_ind_cell.source)
        print(self.run_cell(sel_ind_cell, self.default_index)[1][0])
        #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')

    def preprocess(self, nb, resources=None, km=None):
        """Preprocess notebook executing each code cell. The input
        argument `nb` is modified in-place.

        Parameters
        ----------
        nb : NotebookNode
            Notebook being executed.
        resources : dictionary (optional)
            Additional resources used in the conversion process. For example,
            passing ``{'metadata': {'path': run_path}}`` sets the
            execution path to ``run_path``.
        km: KernelManager (optional)
            Optional kernel manager. If none is provided, a kernel manager will
            be created.

        Returns
        -------
        nb : NotebookNode
            The executed notebook.
        resources : dictionary
            Additional resources used in the conversion process.
        """
        if not resources:
            resources = {}

        with self.setup_preprocessor(nb, resources, km=km):
            # nbconvert overhead
            self.log.info("Executing notebook with kernel: %s" % self.kernel_name)

            # actual preprocessing
            # NOTE: this differs from AnalysisPreprocessor in that
            # we simulate data first
            self.simulate_data(resources)
            nb, resources = ExecutePreprocessor.preprocess(self, 
                                                           nb, 
                                                           resources,
                                                           km=km)

            self.capture_selection_indicators(resources)
            self.capture_sufficient_statistics(resources)

            # nbconvert overhead
            # info_msg = self._wait_for_reply(self.kc.kernel_info())
            # nb.metadata['language_info'] = info_msg['content']['language_info']
            # self.set_widgets_metadata()

        return nb, resources

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
        #print("Preprocessing Cell", cell_index)
        resources.setdefault('fixed_selection', {})
        resources.setdefault('set_selection', {})
        resources.setdefault('data_model', {})
        resources.setdefault('original_data_name', self.original_data_name)

        self.logstr += 'DATA %s\n' % self.data_name
        #self.exec_debug('print(%s)' % self.data_name)
        self.logstr += '\n'

        #ls_cell = nbformat.v4.new_code_cell()
        #ls_cell.source = 'print("NAMESPACE"); ls()'
        #self.logstr += ('listing\n')
        #self.logstr += str(self.run_cell(ls_cell, cell_index)[1])
        #self.logstr += '\n' + '='*40 + '\n'

        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources

        """
        print('SIMULATION CELL SOURCE')
        print('-'*20)
        print(cell.source)
        print('-'*20)
        """
        
        if self.selection_list_name == 'NOTCREATED':
            self.selection_list_name = 'simulated_selection_' + str(uuid.uuid1()).replace('-','') 
            if self.km.kernel_name == 'python3':
                cell.source = ('%s = {};\n' % self.selection_list_name) + cell.source
            elif self.km.kernel_name == 'ir':
                cell.source = ('%s = list();\n' % self.selection_list_name) + cell.source

        _, outputs = self.run_cell(cell, cell_index)
        #self.exec_debug('print("NAMESPACE")'); self.exec_debug('ls()')

        # Capture selection (if applicable)
        self.capture_selection(cell, resources)

        # Recall original selection
        # TODO: implement this
        self.recall_original_selection(cell, resources)

        cell.outputs = outputs
        
        return cell, resources
