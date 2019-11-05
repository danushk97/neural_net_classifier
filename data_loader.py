import pandas as pd
import numpy as np
import preprocess_data as prepro
import h5py

class DataLoader(object):

    def __init__(self, path):
        self.path = path

    def __call__(self):
        data = pd.read_csv(self.path)
        data = self.shuffle_data(data)[0 : 2000]

        if data['y'].dtype == 'object':
            data['y'] = prepro.target_value_encoder(data['y'].copy())


        train_data, valid_data = self.split_data(80, data.copy())
        
        self.train_y = np.array(train_data['y']).reshape(1, -1)
        self.valid_y = np.array(valid_data['y']).reshape(1, -1)
        self.train_x = prepro.read_image(train_data['img_path'])
        self.valid_x = prepro.read_image(valid_data['img_path'])

    def shuffle_data(self, data):

        return data.reindex(np.random.permutation(data.index))

    def split_data(self, train_range, data):
        
        train_number = round(len(data) * (train_range / 100))
        # print(len(data))
        # print(train_number)
        train_data = data[ : train_number]
        valid_data = data[train_number : ]
        print('train data', train_data.shape)
        print('valid data', valid_data.shape)
        return train_data, valid_data
    

def load_cat_vs_non_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    train_set_x_orig = prepro.normalize(train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T)
    test_set_x_orig = prepro.normalize(test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T )
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
