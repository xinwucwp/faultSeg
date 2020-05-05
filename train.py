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
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log', **kwargs):
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
# input image dimensions
params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
seismPathT = "../data/train/seis/"
faultPathT = "../data/train/fault/"
seismPathV = "../data/validation/seis/"
faultPathV = "../data/validation/fault/"

train_ID = []
valid_ID = []
for sfile in os.listdir(seismPathT):
  if sfile.endswith(".dat"):
    train_ID.append(sfile)
for sfile in os.listdir(seismPathV):
  if sfile.endswith(".dat"):
    valid_ID.append(sfile)
print(len(train_ID))
print(len(tests_ID))
train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,data_IDs=train_ID,**params)
valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,data_IDs=valid_ID,**params)

from unet3 import *
model = unet(input_size=(None, None, None,1))
model.compile(optimizer=Adam(lr=1e-3), loss=mymse)
model.summary()

# checkpoint
filepath="check/fseg-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        verbose=1, save_best_only=False, mode='max')
logging = TrainValTensorBoard()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
callbacks_list = [checkpoint, logging, reduce_lr]

print("data prepared, ready to train!")
# Fit the model
history=model.fit_generator(generator=train_generator,validation_data=tests_generator,epochs=100,callbacks=callbacks_list,verbose=1)
model.save('check/fseg.hdf5')
