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
model = load_model('check/fseg-'+'70.hdf5',
                 #'impd.hdf5',
                 custom_objects={
                     'cross_entropy_balanced': cross_entropy_balanced
                 } 
                 )
def main():
  #goTrainTest()
  goValidTest()
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
  gx = gx-gm
  gx = gx/gs
  '''
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  '''
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  gx1 = gx[50,:,:]
  fx1 = fx[50,:,:]
  fp1 = fp[50,:,:]
  gx2 = gx[:,29,:]
  fx2 = fx[:,29,:]
  fp2 = fp[:,29,:]
  gx3 = gx[:,:,29]
  fx3 = fx[:,:,29]
  fp3 = fp[:,:,29]
  plot2d(gx1,fx1,fp1,png='fp1')
  plot2d(gx2,fx2,fp2,png='fp2')
  plot2d(gx3,fx3,fp3,png='fp3')

def goF3Test(): 
  seismPath = "./data/prediction/f3d/"
  n3,n2,n1=512,384,128
  gx = np.fromfile(seismPath+'gxl.dat',dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  '''
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  '''
  gx = np.transpose(gx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
  fp = fp[0,:,:,:,0]
  gx1 = gx[99,:,:]
  fp1 = fp[99,:,:]
  gx2 = gx[:,29,:]
  fp2 = fp[:,29,:]
  gx3 = gx[:,:,29]
  fp3 = fp[:,:,29]
  plot2d(gx1,fp1,fp1,at=1,png='f3d/fp1')
  plot2d(gx2,fp2,fp2,at=2,png='f3d/fp2')
  plot2d(gx3,fp3,fp3,at=2,png='f3d/fp3')

def plot2d(gx,fx,fp,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main()


