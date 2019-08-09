import base64
import feather
import json
import numpy as np
import os
import tempfile
from IPython.display import display

# R code to write a dataframe to a feather and display the raw bytecode
dataframe_to_jsonR = '''
dataframe_to_display <- function(D) {
    filename = tempfile()
    feather::write_feather(D, filename)
    A = readBin(file(filename, 'rb'), 'raw', file.size(filename) + 1000)
    IRdisplay:::display_raw("application/selective.inference", TRUE, list(encoder="feather", data=as.raw(A)), NULL)
}
'''

def base64_to_dataframe(base64_data):
    """Decodes a string of base 64 data into a dataframe. Assumes the
    input string is the raw base 64 encoding of a feather file.

    Parameters
    ----------
    base64_data : string
        A string representing the base 64 encoding of a feather file.

    Returns
    -------
    dataframe : Pandas dataframe
        The dataframe reconstructed from the given base 64 code.
    """
    tmpfile = tempfile.mkstemp()[1]
    byte_str = base64.b64decode(base64_data)  # raw bytecode from base64

    # Write the raw bytecode into a tempfile (i.e. reconstruct the
    # feather file)
    with open(tmpfile, 'wb') as f:
        f.write(byte_str)

    # Read the feather file into a dataframe
    dataframe = feather.read_dataframe(tmpfile)

    return dataframe


def dataframe_to_display(data_frame):
    """Save an array of floats to JSON.
    """
    # Write the array to a temporary file
    filepath = tempfile.mkstemp()[1]
    feather.write_dataframe(data_frame, filepath)

    # Read the temporary file as bytes
    array_data = open(filepath, 'rb').read()
    os.remove(filepath)
    
    # Convert raw bytes to a list of ints
    array_bytes_as_ints = []
    for d in array_data:
        array_bytes_as_ints.append(d)
    
    # Return the JSON representation of the list of ints
    return json.dumps(array_bytes_as_ints)
