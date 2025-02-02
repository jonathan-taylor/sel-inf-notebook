import copy
import time # for random seed
import json
import nbformat
import numpy as np
import uuid
from contextlib import contextmanager
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets import Unicode, Int, Bool, default, Instance
from json_dataframe import base64_to_dataframe

from jupyter_client.manager import start_new_kernel

def _uniq(varname):
    return varname + '_' + str(uuid.uuid1()).replace('-','') 

_importFrom_ir = '''
importFrom  <- function(filename, obj_names, where) {
    if (missing(obj_names)) { ## import everything
    source(filename, local = FALSE)
    } else {
        e  <- new.env()
        return_env = new.env()
        source(filename, local = e)
        #if (missing(where)) where = parent.env(parent.env(e))
        for (obj in obj_names) {
            assign(x = obj, value = get(x = obj, envir = e), envir = return_env)
        }
    }
    invisible(TRUE)
    return(return_env)
}
'''

def _capture_df_python3(dfname):
    #TODO: clean up namespace after running such code? maybe hide imports in a function?
    source = '''
from IPython.display import display
import tempfile, feather,
%(filename)s = tempfile.mkstemp()[1]
feather.write_dataframe(%(dfname)s, %(filename)s)
display({"application/selective.inference":open(%(filename)s, "rb").read()}, metadata={"encoder":"dataframe"}, raw=True)'])
''' % {'dfname':dfname, 'filename':_uniq('filename')}
    return source.strip() + '\n'

def _capture_df_ir(dfname):
    source = '''
library(feather)
%(filename)s = tempfile()
feather::write_feather(%(dfname)s, %(filename)s)
%(file)s = file(%(filename)s, "rb")
%(bin64)s = readBin(%(file)s, "raw", file.size(%(filename)s) + 1000)
IRdisplay:::display_raw("application/selective.inference", TRUE, NULL, %(filename)s, list(encoder="dataframe"))
close(%(file)s)
unlink(%(filename)s)
''' % {'dfname':dfname,
       'file':_uniq('file'),
       'filename':_uniq('filename'),
       'bin64':_uniq('bin64')}
    return source

def _capture_df_local_ir(dfname, listname='scoobydoo'):
    source = '''
%(listname)s[['%(dfname)s']] = rbind(%(listname)s[['%(dfname)s']],%(dfname)s)
''' % {'listname':listname,
       'dfname':dfname}
    return source.strip() + '\n'

def _capture_df_local_python3(dfname, listname='scoobydoo'):
    source = '''
%(listname)s.setdefault('%(dfname)s', []).append(%(dfname)s)
''' % {'listname':listname,
       'dfname':dfname}
    return source.strip() + '\n'

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
    selection_list_name = Unicode('NULL')
    default_index = Int(-1)
    processing_mode = Unicode()
    sufficient_stat_name = Unicode()

    # store the notebook containing all cells we run in preprocessor

    nb_log = Instance(nbformat.notebooknode.NotebookNode)
    nb_log_name = Unicode()
    @default('nb_log')
    def _default_nb_log(self):
        return nbformat.v4.new_notebook()

    def run_cell(self, cell, cell_index, log=True, write=True):
        if self.nb_log is not None and log: 
            self.nb_log.cells.append(cell)
            nbformat.write(self.nb_log, open(self.nb_log_name, 'w'))
        return ExecutePreprocessor.run_cell(self, cell, cell_index)

    def _capture(self, varname, log=False, local=False): # basically up to 2darray
        if not local:
            _capture = {'python3':_capture_df_python3,
                        'ir':_capture_df_ir}[self.km.kernel_name]
        else:
            _capture = lambda df: {'python3':_capture_df_local_python3,
                                   'ir':_capture_df_local_ir}[self.km.kernel_name](df, listname=self.collector)
        result_cell = nbformat.v4.new_code_cell(source=_capture(varname))
        _, cell_output = self.run_cell(result_cell, 0, log=log)
        if not local:
            result_base64 = cell_output[0]['data']['application/selective.inference']
            result_df = base64_to_dataframe(result_base64)
            return result_df

    @default('data_name')
    def _default_data_name(self):
        return _uniq('data')

    define_data = Unicode()
    @default('define_data')
    def _default_define_data(self):
        """Generate source code to define data.
        """
        if self.km.kernel_name == 'python3':
            source = ['if "%(data)s" not in locals():', '    %(data)s = {}']
            source = '\n'.join(source) % {'data':self.data_name}
        elif self.km.kernel_name == 'ir':
            source = _importFrom_ir + '\nif (! exists("%(data)s")) { %(data)s = list() }'
            source = source % {'data':self.data_name}
        return source


    def capture_selection(self, cell, resources, save_to_notebook=False):
        """Capture JSON- or feather/base64-serialized selection outputs
        """
        # Generate a new cell after the cell whose metadata contains
        # the attribute 'capture_selection'

        if 'capture_selection' in cell.metadata:

            capture_cell = nbformat.v4.new_code_cell()

            if self.selection_list_name == 'NULL':
                self.selection_list_name = _uniq('analysis_selection')
                if self.km.kernel_name == 'python3':
                    capture_cell.source = ('%s = {};\n' % self.selection_list_name) + capture_cell.source
                elif self.km.kernel_name == 'ir':
                    capture_cell.source = ('%s = list();\n' % self.selection_list_name) + capture_cell.source

            for selection in cell.metadata['capture_selection']:
                if self.km.kernel_name == 'python3':
                    capture_cell.source += '%s["%s"] = %s;\n' % (self.selection_list_name,
                                                                 selection['name'],
                                                                 selection['name'])
                elif self.km.kernel_name == 'ir':
                    capture_cell.source += '%s[["%s"]] = %s;\n' % (self.selection_list_name,
                                                                   selection['name'],
                                                                   selection['name'])

            _, selection_outputs = self.run_cell(capture_cell, self.default_index)

    def capture_sufficient_statistics(self, resources, local=False):
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
        suff_stat_var = _uniq('suff_stat')

        # [Source code for phantom notebook cell]
        # Compute sufficient statistic and save it to a variable
        if self.km.kernel_name == 'python3':  # source for Python kernel
            source = '%(suff_stat)s = %(suff_stat_map)s(%(selection_list_name)s);'
        elif self.km.kernel_name == 'ir':  # source for R kernel
            source = '%(suff_stat)s = %(suff_stat_map)s(%(selection_list_name)s);'
        # Apply formatting rules to the source code (i.e. fill in
        # variable names)

        self.sufficient_stat_name = suff_stat_var
        data_name = self.data_name
        
        if hasattr(self, 'analysis_selection_list_name'):
            selection_list_name = self.analysis_selection_list_name # for simulation step
        else:
            selection_list_name = self.selection_list_name

        capture_cell.source = source % {'suff_stat': suff_stat_var,
                                        'selection_list_name': selection_list_name,
                                        'suff_stat_map': resources['data_model']['sufficient_statistics'] 
                                        }

        _, cell_output = self.run_cell(capture_cell, self.default_index)
        resources['suff_stat'] = self._capture(suff_stat_var, log=True, local=local)

        return resources

    def random_seed_code(self, cell, modify=True):

        # modify in place to a seed based on clock time!
        if 'analysis_seed' in cell.metadata:

            if not modify:
                seed = int(cell.metadata['analysis_seed'])
            else: 
                seed = int(time.time())

            code_cell = nbformat.v4.new_code_cell()
            if self.km.kernel_name == 'ir':
                code_cell.source = 'set.seed(%d)' % seed
            elif self.km.kernel_name == 'python3':
                code_cell.source = 'np.random.seed(%d)' % seed
            return code_cell
        else:
            return None


class AnalysisPreprocessor(SelectiveInferencePreprocessor):
    """This preprocessor runs the analysis on the collected data,
    capturing output that will be conditioned on.
    """
    force_raise_errors = True
    cell_allows_errors = False
    processing_mode = Unicode('analysis')
    modify_seed = False

    def preprocess(self, nb, resources=None, km=None, copy_nb=True):
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

        if copy_nb:
            self.nb_log = copy.copy(nb)
            self.nb_log.cells = []

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
        self.append_result_code(resources)

        if self.nb_log_name is not None:
            nbformat.write(self.nb_log, open(self.nb_log_name, 'w'))
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

        code_cell = self.insert_data_input_code(cell)
        if code_cell is not None:
            self.run_cell(code_cell, cell_index)

        seed_cell = self.random_seed_code(cell, modify=self.modify_seed)
        if seed_cell is not None:
            self.run_cell(seed_cell, cell_index)

        _, outputs = self.run_cell(cell, cell_index)

        # Capture selection
        if 'data_model' in cell.metadata and 'functions' in cell.metadata:
            for var in cell.metadata['data_model'].keys():
                resources['data_model'][var] = cell.metadata['data_model'][var] # hooks used to define target and generative model

        self.capture_selection(cell, resources, True)
        cell.outputs = outputs

        return cell, resources

    def datafile_input_source(self, filename, variables):
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
            raise NotImplementedError
        elif self.km.kernel_name == 'ir':
            source = [
                '%(variable_name)s = c(%(variables)s)',
                '%(env)s = importFrom("%(filename)s", %(variable_name)s)',
                'for (%(var)s in %(variable_name)s) { assign(%(var)s, get(%(var)s, env=%(env)s))}'
                ]
            source = '\n'.join(source)
            source = source % {'variable_name':_uniq('variable'),
                               'env':self.data_name,
                               'var':_uniq('var'),
                               'filename': filename,
                               'variables': ', '.join(['"%s"' % v for v in variables])
                               }
        return source


    def insert_data_input_code(self, cell):
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
            code_cell = nbformat.v4.new_code_cell()
            code_cell.source = self.define_data + '\n'
            filename, variables = cell.metadata['data_input']
            code_cell.source += self.datafile_input_source(filename, variables) + '\n'
            return code_cell
        else:
            return None

    def append_result_code(self, resources):
        #TODO allow multivariate estimates
        result_cell = nbformat.v4.new_code_cell()
        template_dict = {'result': _uniq('result'),
                         'estimators':resources['data_model']['estimators'],
                         'variance_map':resources['data_model']['variances'],
                         'analysis_selection':self.selection_list_name,
                         'selection':self.selection_list_name,
                         'suff_stat':self.sufficient_stat_name,
                         'val':_uniq('val'),
                         'names':_uniq('names'),
                         'indicator': _uniq('indicator'),
                         'estimates':_uniq('estimates'),
                         'variance':_uniq('variance'),
                         }
        
        source = '%(result)s = %(estimators)s(%(analysis_selection)s)\n' % template_dict
        source += '%(variance)s = %(variance_map)s(%(analysis_selection)s)' % template_dict

        if self.km.kernel_name == 'ir':
            source += '''
%(estimates)s = c();
%(names)s = c();
for (%(val)s in %(result)s) {
    %(names)s = c(%(names)s, %(val)s[['identifier']])
    %(estimates)s = c(%(estimates)s, %(val)s[['value']])
}
names(%(estimates)s) = %(names)s
%(estimates)s = as.data.frame(as.matrix(t(%(estimates)s)))
''' % template_dict
        elif self.km.kernel_name == 'python3':
            source += ('''
%(names)s = []
%(estimates)s = []
for %(val) in %(result)s.items():
    %(names)s.append(%(val)s['identifier'])
    %(estimates)s.append(%(val)s['value'])
%(estimates)s = pd.DataFrame([%(estimates)s], columns=%(names)s)
''' % template_dict).strip()

        result_cell.source = source
        _, cell_output = self.run_cell(result_cell, 0)

        resources['estimates'] = self._capture('%(estimates)s' % template_dict, log=True)

        variances = []
        cross = []
        for i in range(1, len(resources['estimates'].columns)+1):
            template_dict['idx'] = i
            if self.km.kernel_name == 'ir':
                variances.append(np.asarray(self._capture('as.data.frame(%(variance)s[[%(idx)d]][["var"]])' % template_dict, log=True)))
                cross.append(np.asarray(self._capture('as.data.frame(%(variance)s[[%(idx)d]][["cross"]])' % template_dict, log=True)))
            elif self.km.kernel_name == 'python3':
                variances.append(np.asarray(self._capture('pd.DataFrame(%(variance)s[[%(idx)d]][["variance"]])' % template_dict, log=True)))
                cross.append(np.asarray(self._capture('pd.DataFrame(%(variance)s[[%(idx)d]][["cross"]])' % template_dict, log=True)))

        resources['variances'] = dict(zip(resources['estimates'].columns,
                                          variances))

        resources['cross'] = dict(zip(resources['estimates'].columns,
                                      cross))

class SimulatePreprocessor(SelectiveInferencePreprocessor):
    """This preprocessor reruns the analysis several times on simulated
    data (using a simulation procedure specified by the user). Each
    time, it captures the observed values of variables on which the
    analysis is conditioned.
    """

    collector = Unicode('NULL')
    processing_mode = Unicode('simulation')
    selection_list_name = Unicode('NULL')
    analysis_selection_list_name = Unicode()
    modify_seed = True
    indicator_name = Unicode()
    analysis_data_name = Unicode()

    @default('indicator_name')
    def _default_indicator(self):
        return _uniq('indicator')

    @default('data_name')
    def _default_data_name(self):
        return _uniq('simulated_data')


    def simulate_data(self, resources):
        """Simulate data within the client notebook by injecting a cell
        that calls the simulate function.
        """
        #print('simulating data')
        if self.km.kernel_name == 'python3':
            source = '%(simulated_data)s = %(simulate)s(%(data_name)s, %(analysis_selection)s)' + '\n'
            source += '\n'.join(['for key in %(simulated_data)s.keys():',
                    '    locals()[key] = %(simulated_data)s[key]'])
        elif self.km.kernel_name == 'ir':
            source = '%(simulated_data)s = %(simulate)s(%(data_name)s, %(analysis_selection)s)'
            source += '\n'.join(['\nfor(%(key)s in names(%(simulated_data)s)) {',
                                 '  assign(%(key)s, get(%(key)s, env=%(simulated_data)s))',
                                 '}'])

        source = source % {'simulated_data':self.data_name,
                           'simulate':resources.get('data_model', {}).get('resample_data', 'function_not_found'),
                           'data_name':self.analysis_data_name,
                           'analysis_selection':self.analysis_selection_list_name,
                           'key':_uniq('key')
        }
        
        simulate_cell = nbformat.v4.new_code_cell(source=source)
        """
        print('SIMULATE DATA SOURCE simulate_data')
        print('-'*20)
        print(simulate_cell.source)
        print('-'*20)
        """
        self.run_cell(simulate_cell, self.default_index)

    def preprocess(self, nb, resources=None, km=None, copy_nb=True):
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

        if copy_nb:
            self.nb_log = copy.copy(nb)
            self.nb_log.cells = []

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

            self.capture_sufficient_statistics(resources, local=True)

            # nbconvert overhead
            # info_msg = self._wait_for_reply(self.kc.kernel_info())
            # nb.metadata['language_info'] = info_msg['content']['language_info']
            # self.set_widgets_metadata()

        self.append_result_code(resources)
        if self.nb_log_name is not None:
            nbformat.write(self.nb_log, open(self.nb_log_name, 'w'))
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

        if self.collector == 'NULL':
            self.collector = _uniq('collector')
            collector_cell = nbformat.v4.new_code_cell()
            if self.km.kernel_name == 'ir':
                collector_cell.source = '\nif (! exists("%(collector)s")) { %(collector)s = list() }' % {'collector':self.collector}
            elif self.km.kernel_name == 'python3':
                source = ['if "%(collector)s" not in locals():', '    %(collector)s = {}']
                source = '\n'.join(source) % {'collector':self.collector}
                collector_cell.source = source
            self.run_cell(collector_cell, 0, log=True)

        resources.setdefault('data_model', {})
        resources.setdefault('analysis_data_name', self.analysis_data_name)

        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources
        
        if self.selection_list_name == 'NULL':
            selection_cell = nbformat.v4.new_code_cell()
            self.selection_list_name = _uniq('simulated_selection')
            if self.km.kernel_name == 'python3':
                selection_cell.source = ('%s = {};\n' % self.selection_list_name)
            elif self.km.kernel_name == 'ir':
                selection_cell.source = ('%s = list();\n' % self.selection_list_name) 
            self.run_cell(selection_cell, cell_index)

        seed_cell = self.random_seed_code(cell, modify=self.modify_seed)
        if seed_cell is not None:
            self.run_cell(seed_cell, cell_index)

        _, outputs = self.run_cell(cell, cell_index)

        # Capture selection (if applicable)
        self.capture_selection(cell, resources)

        cell.outputs = outputs
        
        return cell, resources

    def append_result_code(self, resources):
        result_cell = nbformat.v4.new_code_cell()
        template_dict = {'result': _uniq('result'),
                         'estimators':resources['data_model']['estimators'],
                         'analysis_selection':self.analysis_selection_list_name,
                         'selection':self.selection_list_name,
                         'suff_stat':self.sufficient_stat_name,
                         'val':_uniq('val'),
                         'names':_uniq('names'),
                         'indicator': self.indicator_name,
                         'estimates':_uniq('estimates'),
                         }
        
        source = '%(result)s = %(estimators)s(%(analysis_selection)s)' % template_dict
        
        if self.km.kernel_name == 'ir':
            source += '''
%(indicator)s = c();
%(names)s = c();
for (%(val)s in %(result)s) {
    %(names)s = c(%(names)s, %(val)s[['identifier']])
    %(indicator)s = c(%(indicator)s, %(val)s[['indicator']](%(selection)s))
}
names(%(indicator)s) = %(names)s
%(indicator)s = as.data.frame(as.matrix(t(%(indicator)s)))
''' % template_dict
        elif self.km.kernel_name == 'python3':
            source += ('''
%(indicator)s = []
%(names)s = []
for %(val) in %(result)s.items():
    %(names)s.append(%(val)s['identifier'])
    %(indicator)s.append(%(val)s['indicator'](%(selection)s))
    %(estimates)s.append(%(val)s['value'])
%(indicator)s = pd.DataFrame([%(indicator)s], columns=%(names)s)
''' % template_dict).strip()

        result_cell.source = source
        _, cell_output = self.run_cell(result_cell, 0)

        resources['indicators'] = self._capture('%(indicator)s' % template_dict, log=True, local=True)

def inference(resources,
              fit_probability,
              fit_args={},
              alpha=0.1):

    from selectinf.learning.core import _inference
    import functools
    learning_T = np.asarray(resources['sim_suff_stats'])
    learning_Y = np.asarray(resources['sim_indicators'])
    weight_fns = fit_probability(learning_T, learning_Y, **fit_args)

    results = []
    observed_T = np.squeeze(resources['observed_suff_stat'])
    for i in range(len(resources['estimates'])):
        estname = resources['estimates'].columns[i]
        cross = resources['cross'][estname]
        var = resources['variances'][estname]
        estimate = np.squeeze(resources['estimates'][estname])
        if 'hypothesis' in resources:
            hypothesis = resources['hypothesis'][estname]
        else:
            hypothesis = 0

        direction = np.atleast_1d(np.squeeze(cross.dot(np.linalg.inv(var))))
        cur_nuisance = np.atleast_1d(np.squeeze(observed_T - direction * estimate))
        print(direction, cur_nuisance, 'nuisance')

        def new_weight_fn(cur_nuisance, direction, weight_fn, target_val):
            return weight_fn(np.multiply.outer(target_val, direction) + cur_nuisance[None, :])

        new_weight_fn = functools.partial(new_weight_fn, cur_nuisance, direction, weight_fns[i])
        results.append(_inference(estimate,
                                  var.reshape((1, 1)),
                                  new_weight_fn,
                                  hypothesis=hypothesis,
                                  alpha=alpha)[:4])
    return results
