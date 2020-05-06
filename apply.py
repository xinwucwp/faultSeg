import math
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
#from keras import backend
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr
from unet3 import cross_entropy_balanced
import os
pngDir = './png/'
model = load_model('check/fseg-'+'03.hdf5',
                 #'impd.hdf5',
                 custom_objects={
                     'cross_entropy_balanced': cross_entropy_balanced
                 } 
                 )
def main():
  #goTrainTest()
  #goValidTest()
  goF3Test()

def goTrainTest():
  seismPath = "./data/train/seis/"
  faultPath = "./data/train/fault/"
  n1,n2,n3=128,128,128
  dk = 100
  gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
  fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
  gx = np.reshape(gx,(n1,n2,n3))
  fx = np.reshape(fx,(n1,n2,n3))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  gx1 = gx[50,:,:]
  fx1 = fx[50,:,:]
  fp1 = fp[50,:,:]
  plot2d(gx1,fx1,fp1,png='fp')

def goValidTest():
  seismPath = "./data/validation/seis/"
  faultPath = "./data/validation/fault/"
  n1,n2,n3=128,128,128
  dk = 2
  gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
  fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
  gx = np.reshape(gx,(n1,n2,n3))
  fx = np.reshape(fx,(n1,n2,n3))
  gm = np.mean(gx)
  gs = np.std(gx)
  #gx = gx-gm
  #gx = gx/gs
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  gx1 = gx[50,:,:]
  fx1 = fx[50,:,:]
  fp1 = fp[50,:,:]
  plot2d(gx1,fx1,fp1,png='fp')

def goF3Test(): 
  seismPath = "./data/prediction/f3d/"
  n3,n2,n1=512,384,128
  gx = np.fromfile(seismPath+'gxl.dat',dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  gx = gx*255
  gx = np.transpose(gx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  gx1 = gx[99,:,:]
  fp1 = fp[99,:,:]
  plot2d(gx1,fp1,fp1,png='f3d/fp')

def plot2d(gx,fx,fp,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  gmin = np.min(gx)
  gmax = np.max(gx)
  ax.imshow(gx,vmin=gmin,vmax=gmax,cmap=plt.cm.bone,interpolation='bicubic',aspect=1)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=1)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=1)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main()


