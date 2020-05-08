import os
import glob
import argparse
import matplotlib
from PIL import Image
import numpy as np
import csv
#from skimage.io import imsave



# Keras / TensorFlow
from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict, load_images, display_images
from matplotlib import pyplot as plt

def denseDepthModel( model, labelInfo):
  # Input images
  with open( labelInfo, 'r') as labelData:
    labels = csv.reader(labelData, delimiter=';')
    
    image_names   = []
    loaded_images = []
    for labelID,label in enumerate(labels):
      if labelID%1000 == 0 and labelID != 0:
        
        inputs = np.stack(loaded_images, axis=0)
        # Compute results
        outputs = predict(model, inputs, batch_size=1000)
        # Save results
        for i in range(outputs.shape[0]):
          #imsave(opDir+'sci_'+ip_names[i],outputs[i][:,:,0])
          rescaled = outputs[i][:,:,0]
          rescaled = rescaled - np.min(rescaled)
          rescaled = rescaled * 255 / np.max(rescaled)
          img = Image.fromarray(np.uint8(rescaled), mode='L').resize((224,224), Image.ANTIALIAS)
          img.save(image_names[i], quality=85, optimize=True)
        
        image_names   = []
        loaded_images = []
      
      x = np.clip(np.asarray(Image.open(label[1]).resize((640,480), Image.ANTIALIAS), dtype=float) / 255, 0, 1)
      loaded_images.append(x)
      image_names.append(label[3])
      
    if len(image_names) != 0:
      inputs = np.stack(loaded_images, axis=0)
      # Compute results
      outputs = predict(model, inputs, batch_size=len(image_names))
      # Save results
      for i in range(outputs.shape[0]):
        #imsave(opDir+'sci_'+ip_names[i],outputs[i][:,:,0])
        rescaled = outputs[i][:,:,0]
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled * 255 / np.max(rescaled)
        img = Image.fromarray(np.uint8(rescaled), mode='L').resize((224,224), Image.ANTIALIAS)
        img.save(image_names[i], quality=85, optimize=True)
        
      
    

    
