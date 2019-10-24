"""Inference after Black Box Selection

Code by Alex Kim for Prof. Jonathan Taylor
Python 3.5.3

This module uses our custom nbconvert preprocessors to parse a Jupyter
Notebook and extract information for inference.
"""
import argparse
import nbformat
import numpy as np

import preprocessors

def analyze(nb, resources={}):
    """Run the analysis preprocessor on a given notebook.
    """
    print('\nAnalyzing...')

    # Initialize a preprocessor and preprocess the notebook
    preprocessor = preprocessors.AnalysisPreprocessor(timeout=600)
    nb, resources = preprocessor.preprocess(nb, resources=resources)

    # Extract the sufficient statistic and selected variables
    results = get_results(resources)
    print(results['suf_stat'])
    print(results['sel_vars'])
    print('Analysis complete.')

    return (nb, resources, preprocessor, results)


def simulate(nb, resources, preprocessor_init, n):
    """Run analysis on simulated or resampled data in the given
    notebook.

    Parameters
    ----------
    nb : NotebookNode
    resources : dictionary
    preprocessor :
    n : int
        The number of times to simulate and analyze.

    Returns
    -------
    """
    print('\nSimulating...')

    # Initialize the simulation preprocessor (for simulated data)
    preprocessor = preprocessors.SimulatePreprocessor(timeout=600,
            km=preprocessor_init.km)  # pass in the existing kernel manager
    preprocessor.data_name = preprocessor_init.data_name

    # Simulate n times, save the suf stat and sel vars to tables
    for i in range(n):
        print('Simulation %s of %s' % (i+1, n))

        # Preprocess and save results
        nb, resources = preprocessor.preprocess(nb, resources=resources,
                                                km=preprocessor.km)
        sim_results = get_results(resources)

        if i == 0:
            suf_stat_table = sim_results['suf_stat']
            sel_vars_table = sim_results['sel_vars']
        else:
            suf_stat_table = np.append(suf_stat_table,
                                       sim_results['suf_stat'], axis=0)
            sel_vars_table = np.append(sel_vars_table,
                                       sim_results['sel_vars'], axis=0)

        print(suf_stat_table)
        print(sel_vars_table)

    results = {'suf_stat': suf_stat_table, 'sel_vars': sel_vars_table}
        
    # Shut down the kernel manager
    preprocessor.kc.stop_channels()
    preprocessor.km.shutdown_kernel(now=preprocessor.shutdown_kernel == 'immediate')
    print('Simulation complete. Performed %s runs.' % n)

    return (nb, resources, results)


def get_results(resources):
    """Extract the sufficient statistic and selected variables from a
    given resources dictionary. Designed for both analysis and
    simulation.
    """
    # Determine the selection type (fixed vs set)
    # TODO: perhaps simplify the selection type logic (see TODO file)
    selection_type = resources['selection_type']
    if selection_type == 'set':
        selection_var = 'set_selection'
    elif selection_type == 'fixed':
        selection_var = 'fixed_selection'
    else:
        # TODO: turn this into an exception
        print('WARNING: Unspecified selection type')

    # Save selection and sufficient statistic
    suf_stat = np.array(resources['suff_stat'])
    suf_stat = np.transpose(suf_stat)
    sel_vars = np.array(resources[selection_var]['selected_vars'])
    sel_vars = np.transpose(sel_vars)
    results = {'suf_stat': suf_stat, 'sel_vars': sel_vars}

    return results


def get_fixed_sel_indicators(sel_vars_init, sel_vars_sim):
    n = np.shape(sel_vars_sim)[0]
    indicators = np.empty(n)
    for i in range(n):
        indicators[i] = np.array_equal(sel_vars_sim[i], sel_vars_init)
    return indicators


def get_set_sel_indicators(sel_vars_init, sel_vars_sim):
    n_sim = np.shape(sel_vars_sim)[0]
    n_vars = sel_vars_init.shape[1]
    indicators = np.zeros((n_sim, n_vars))
    for i in range(n_sim):
        for j in range(n_vars):
            if np.isin(sel_vars_init[0][j], sel_vars_sim[i]):
                indicators[i][j] = 1
    return indicators


if __name__ == "__main__":
    """Read in a notebook from the given CLI argument, then preprocess
    it to extract relevant information, then perform inference.
    """
    print("\n-- SELECTIVE INFERENCE --")

    # Read in the notebook as a dict-like object for preprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook_path')
    nbpath = parser.parse_args().notebook_path
    nb = nbformat.read(nbpath, nbformat.NO_CONVERT)
    print("Client Notebook:", nbpath)

    # Perform initial analysis on the notebook
    nb, resources, preprocessor, results_init = analyze(nb)
    suf_stat_init = results_init['suf_stat']
    sel_vars_init = results_init['sel_vars']

    # Perform analysis on simulated data
    nb, resources, results_sim = simulate(nb, resources, preprocessor, 3)
    suf_stat_sim = results_sim['suf_stat']
    sel_vars_sim = results_sim['sel_vars']

    # Get the selection indicators
    selection_type = resources['selection_type']
    #selection_type = 'full'
    n_simulations = np.shape(sel_vars_sim)[0]
    if selection_type == 'full':
        indicators = get_fixed_sel_indicators(sel_vars_init, sel_vars_sim)
    elif selection_type == 'set':
        indicators = get_set_sel_indicators(sel_vars_init, sel_vars_sim)

    # Print final results
    print('\n-- FINAL RESULTS --')

    print('\nSufficient Statistic for Original Data:')
    print(suf_stat_init)

    print('\nSufficient Statistics for Simulated Data:')
    print(suf_stat_sim)

    print('\nSelection Indicators for Simulated Data:')
    print(indicators)
