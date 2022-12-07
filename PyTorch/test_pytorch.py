import os
import sys
import glob
import argparse
import matplotlib
import numpy as np

sys.path.insert(0,"../")
sys.path.insert(1,"./")
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import load_images
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from model_pt import PTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='../nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='../examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--cuda', default=1, type=int, help='Enable of Disbale Cuda')
args = parser.parse_args()
if args.cuda==0:
  device = 'cpu'

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

keras_name = []
for name, weight in zip(names, weights):
  keras_name.append(name)

pytorch_model = PTModel().float()

# load parameter from keras
keras_state_dict = {} 
j = 0
for name, param in pytorch_model.named_parameters():
  
  if 'classifier' in name:
    keras_state_dict[name]=param
    continue

  if 'conv' in name and 'weight' in name:
    keras_state_dict[name]=torch.from_numpy(np.transpose(weights[j],(3, 2, 0, 1)))
    # print(name,keras_name[j])
    j = j+1
    continue
  
  if 'conv' in name and 'bias' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    continue

  if 'norm' in name and 'weight' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].shape)
    j = j+1
    continue

  if 'norm' in name and 'bias' in name:
    keras_state_dict[name]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    keras_state_dict[name.replace("bias", "running_mean")]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    keras_state_dict[name.replace("bias", "running_var")]=torch.from_numpy(weights[j])
    # print(param.shape,weights[j].size)
    j = j+1
    continue


pytorch_model.load_state_dict(keras_state_dict)
# pytorch_model = torch.load('depth_3.pth')
pytorch_model.eval()
#torch.save(pytorch_model,"depth_3.pth")
pytorch_model.to(device)
if device.__eq__('cuda'):
  print("Loaded model to GPU")
else: 
  print("Loaded model to CPU")


def my_DepthNorm(x, maxDepth):
    return maxDepth / x

def my_predict(model, images, minDepth=10, maxDepth=1000):

  with torch.no_grad():
    # Compute predictions
    predictions = model(images.to(device))

    # Put in expected range
  return np.clip(my_DepthNorm(predictions.cpu().numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
import time
# # Input images
inputs = load_images( glob.glob(args.input) ).astype('float32')

pytorch_input = torch.from_numpy(inputs[0,:,:,:]).permute(2,0,1).unsqueeze(0)

print("Input Shape = " + str(pytorch_input.shape))
# print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results (When it prdeicts on first it takes some time after that it runs fast you can check with using for loop)
for i in range(10):
  tic = time.time()
  output = my_predict(pytorch_model,pytorch_input[0,:,:,:].unsqueeze(0))
  toc  = time.time()
  print("Time for test "+str(i)+" "+str(1000*(toc-tic))+" ms")

print("Output Shape = " + str(output.shape))
plt.imshow(output[0,0,:,:])
plt.savefig('test.png')
plt.show()
