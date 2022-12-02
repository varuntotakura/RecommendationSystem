import os
import cv2
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.layers import Input
from keras.backend import reshape
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from Models.CNN_Model import Xception

def getImagePaths(path):
    """
    Function to Combine Directory Path with individual Image Paths
    
    parameters: path(string) - Path of directory
    returns: image_names(string) - Full Image Path
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def preprocess_img(img_path):
    dsize = (225,225)
    new_image=cv2.imread(img_path)
    new_image=cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)  
    new_image=np.expand_dims(new_image,axis=0)
    new_image=preprocess_input(new_image)
    return new_image

def load_data(images_dir):
    output = []
    output = getImagePaths(images_dir)[:10000]
    return output

def model():
    model = Xception(weights = 'imagenet', include_top=False)
    for layer in model.layers:
        layer.trainable=False
        #model.summary()
    return model

def feature_extraction(image_data, model):
    features=model.predict(image_data)
    features=np.array(features)
    features=features.flatten()
    return features

def result_vector_cosine(model,feature_vector,new_img):
    new_feature = model.predict(new_img)
    new_feature = np.array(new_feature)
    new_feature = new_feature.flatten()
    N_result = 12
    knn = NearestNeighbors(n_neighbors = N_result, metric="cosine").fit(feature_vector)
    distances, indices = knn.kneighbors([new_feature])
    return(indices)

def input_show(data):
    plt.title("Query Image")
    plt.imshow(data)
    plt.show()
    plt.savefig('./Results/Query.png')
  
def show_result(data,result):
    fig = plt.figure(figsize=(9,8))
    for i in range(4,12):
        index_result=result[0][i]
        plt.subplot(3,4,i+1)
        plt.imshow(cv2.imread(data[index_result]))
    plt.show()
    plt.savefig('./Results/QueryResult.png')

def main(imageval):
    images_dir = '../input/h-and-m-personalized-fashion-recommendations/images'
    features=[]
    output = load_data(images_dir)
    main_model = model()
    #Limiting the data for training
    for i in output[:999]:
        new_img = preprocess_img(i)
        features.append(feature_extraction(new_img, main_model))
    feature_vec = np.array(features)
    result=result_vector_cosine(main_model,feature_vec, preprocess_img(output[imageval]))
    input_show(cv2.imread(output[imageval]))
    show_result(output, result)
    output = load_data(images_dir)

if __name__=='__main__':
    main(99)