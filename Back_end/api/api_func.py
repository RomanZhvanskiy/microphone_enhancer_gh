import os

def list_path(path_to_list:str):
    """
    Lists files at specific path
        Parameters:
        path (str): Path where list of files needs to be onbtained from.
    """
    out = []
    for filename in os.listdir(path_to_list):
        out.append({
            "name": filename.split(".")[0],
            "path": os.path.join(path_to_list, filename)
        })
    return out
