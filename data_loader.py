import pandas as pd
import numpy as np
import preprocess_data as prepro

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
        print(len(data))
        print(train_number)
        train_data = data[ : train_number]
        valid_data = data[train_number : ]

        return train_data, valid_data
