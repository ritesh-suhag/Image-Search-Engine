
#########################################################################
# Convolutional Neural Network - Image Search Engine
#########################################################################


###########################################################################################
# import packages
###########################################################################################

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

###########################################################################################
# bring in pre-trained model (excluding top)
###########################################################################################

# image parameters

img_width = 224
img_height = 224
num_channels = 3

# network architecture

vgg = VGG16(input_shape = (img_width, img_height, num_channels),
            include_top = False,
            pooling = 'avg')

model = Model(inputs = vgg.input,
              outputs = vgg.layers[-1].output)

# save model file

model.save('models/vgg16_search_engine.h5')


###########################################################################################
# preprocessing & featurising functions
###########################################################################################





###########################################################################################
# featurise base images
###########################################################################################

# source directory for base images



# empty objects to append to



# pass in & featurise base image set



# save key objects for future use



        
###########################################################################################
# pass in new image, and return similar images
###########################################################################################

# load in required objects



# search parameters


        
# preprocess & featurise search image


        
# instantiate nearest neighbours logic



# apply to our feature vector store



# return search results for search image (distances & indices)



# convert closest image indices & distances to lists



# get list of filenames for search results



# plot results

plt.figure(figsize=(12,9))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





