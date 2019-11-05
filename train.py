import pickle

from data_loader import load_cat_vs_non_cat_dataset
from classifier import neural_net_classifier

def train(data_path, model_file):
    # data_loader = DataLoader(data_path)
    train_x, train_y, test_x, test_y, classes = load_cat_vs_non_cat_dataset()

    # weights = load_weights(model_file) if model_file else None
    units_size = [train_x.shape[0], 7, 1]
    classifier = neural_net_classifier(x = train_x,
                                       y = train_y,
                                       units_size = units_size,
                                       threshold = 0.4,
                                       learning_rate = 0.06)
    classifier(epoch = 2500)
    classifier.predict(train_x, train_y)
    classifier.predict(test_x, test_y)
    save_model(classifier)

def save_model(model):
    pkl_representation = pickle.dumps(model)
    
    with open('model', 'wb') as file:
         
        pickle.dump(pkl_representation, file) 

def load_weights(path):
    pkl_obj = None
    model_obj = None

    with open(path, 'rb') as file:
        pkl_obj = pickle.load(file)

    if pkl_obj:
        model_obj = pickle.loads(pkl_obj)

    return model_obj.parameters

if __name__=='__main__':
    train(data_path = 'train_data.csv', model_file = 'model')
