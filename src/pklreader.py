import pickle

def workspace_file_loader(path_name):
  with open(path_name, 'rb') as f:
      data = pickle.load(f)
      return data