import os
import glob
import argparse
import matplotlib
from PIL import Image
import numpy as np
#from skimage.io import imsave



# Keras / TensorFlow
from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict, load_images, display_images
from matplotlib import pyplot as plt

def denseDepthModel( model, ipDir, opDir):
  # Input images
  inputs, ip_names = load_images( glob.glob(ipDir) )
  #print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

  # Compute results
  outputs = predict(model, inputs)

  #matplotlib problem on ubuntu terminal fix
  #matplotlib.use('TkAgg')   

  # Save results
  for i in range(outputs.shape[0]):
    #imsave(opDir+'sci_'+ip_names[i],outputs[i][:,:,0])
    rescaled = outputs[i][:,:,0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled * 255 / np.max(rescaled)
    img = Image.fromarray(np.uint8(rescaled), mode='L').resize((200,200), Image.ANTIALIAS)
    img.save(opDir+ip_names[i], , quality=50, optimize=True)
