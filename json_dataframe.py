import tempfile, os, json
import numpy as np
import feather
from IPython.display import display

dataframe_to_jsonR = '''
dataframe_to_display = function(D) {
    filename = tempfile()
    feather::write_feather(D, filename)
    A = readBin(file(filename, 'rb'), 'raw', file.size(filename) + 1000)
    IRdisplay:::display_raw("application/selective.inference", TRUE, list(encoder="feather", data=as.raw(A)), NULL)
}
'''

def base64_to_dataframe(base64_data):
    byte_str = base64.b64decode(base64_data)
    tmpfile = tempfile.mkstemp()[1]
    with open(tmpfile, 'wb') as f:
        f.write(byte_str)
    return feather.read_dataframe(tmpfile)

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



