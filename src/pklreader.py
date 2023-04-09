import pickle


# ' A reader to load pickle files.
# '
# ' @param path_name: The path of the pickle file.
# '
# ' @return Return a decoded pickle file.
def workspace_file_loader(path_name):
  with open(path_name, 'rb') as f:
      data = pickle.load(f)
      return data