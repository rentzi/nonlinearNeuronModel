import pickle
import os


def saveVar(var, file_path):
    """
    save_var saves a var to a specified filePath
    INPUT:
    var: the variable to be saved
    file_path: the filepath you want to save the var, for example data/.../var.pckl. If the path does not exist, it creates it
    """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):  # makes the directory if it does not exist
        os.makedirs(directory)

    # uses pickle to serialize and save the variable var
    pickle_out = open(file_path, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()



def loadVar(file_path):
    """
    LOADVAR loads a var from a specified filePath
    INPUT:
    filePath: where the variable is
    OUTPUT:
    var: the variable loaded
    """

    pickle_in = open(file_path, "rb")
    var = pickle.load(pickle_in)
    pickle_in.close()

    return var   