from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)
import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from utils import DataGenerator
from unet3 import *

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"

  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  train_ID = range(200)
  valid_ID = range(20)
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  model = unet(input_size=(None, None, None,1))
  model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
                metrics=['accuracy'])
  model.summary()

  # checkpoint
  filepath="check1/fseg-{epoch:02d}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        verbose=1, save_best_only=False, mode='max')
  logging = TrainValTensorBoard()
  #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
  #                              patience=20, min_lr=1e-8)
  callbacks_list = [checkpoint, logging]
  print("data prepared, ready to train!")
  # Fit the model
  history=model.fit_generator(generator=train_generator,
  validation_data=valid_generator,epochs=100,callbacks=callbacks_list,verbose=1)
  model.save('check1/fseg.hdf5')
  showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

  # summarize history for loss
  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log1', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        logs.update({'lr': keras.eval(self.model.optimizer.lr)})
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':
    main()

