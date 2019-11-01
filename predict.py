import pickle

with open('model', 'rb') as file:
    model = pickle.load(file)
    print(model)