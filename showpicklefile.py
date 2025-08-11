import pickle

# Specify the path to your .pickle file
file_path = 'D:\PTN Robot\AICam\data.pickle'

try:
    # Open the file in binary read mode ('rb')
    with open(file_path, 'rb') as f:
        # Load the pickled object
        data = pickle.load(f)

    # Now 'data' holds the Python object that was saved in the pickle file.
    # You can print it to see its contents.
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")