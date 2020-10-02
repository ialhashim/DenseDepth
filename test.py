import os
import glob
import argparse
import matplotlib
from PIL import Image
import numpy as np
from tqdm import tqdm

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

import os
x = os.listdir('/content/DenseDepth/examples')
for i in tqdm(range(len(x))):
  try:
    inputs = load_images(glob.glob('examples/'+ x[i]))
  except:
    continue

  outputs = predict(model, inputs)
  import matplotlib.pyplot as plt
  import numpy as np
  for i in (range(len(outputs))):
    rescaled = outputs[i][:,:,0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled / np.max(rescaled)


    plasma = plt.get_cmap('plasma')
    y = Image.fromarray(np.uint8(plasma(rescaled)*255))
    y.save('/content/blargh/{}.png'.format(x[i].split()[0]))

'''
#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
#viz = display_images(outputs.copy(), inputs.copy())
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.savefig('test.png')
#plt.show()
'''
