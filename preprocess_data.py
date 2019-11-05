import numpy as np
import cv2 

def target_value_encoder(y):
    to_category = y.astype('category')
    to_category = to_category.cat.codes 

    return to_category

def read_image(data):
    images = []
    
    for d in data:
        img = cv2.imread(d)
        img = cv2.resize(img, (150, 150))
        images.append(img)
    
    images = np.array(images, dtype = float) 
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))
    images = nprmalize(images.T)
    print(images.shape)
    
    return images

def normalize(img):

    return img / 255
