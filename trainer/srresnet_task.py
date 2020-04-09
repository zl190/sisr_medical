"""
Copyright Zisheng Liang 2019 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from trainer import utils, callbacks, config
from trainer.datasets import deeplesion_lr_hr_pair
from trainer.models.sisr import MySRResNet
import os
from pathlib import Path


_DATA_DIR = '/datacommons/plusds/team10-zl190/Spring20/tensorflow_datasets' # data_dir


# dataset
train_dataset, train_count = deeplesion_lr_hr_pair(split='train', 
                                                   size=(config.im_h, config.im_w, 1), 
                                                   downsampling_factor=config.upsampling_rate, 
                                                   batch_size=config.batch_size, 
                                                   augment=False, 
                                                   data_dir = config.data_dir)
validation_dataset, validation_count = deeplesion_lr_hr_pair(split='validation', 
                                                   size=(config.im_h, config.im_w, 1), 
                                                   downsampling_factor=config.upsampling_rate, 
                                                   batch_size=config.batch_size, 
                                                   augment=False, 
                                                   data_dir = config.data_dir)

# model
if config.g_weight:
  model = tf.keras.models.load_model(config.g_weight, custom_objects={"ssim": utils.ssim, "psnr": utils.psnr})
else:
  model = MySRResNet(shape=(config.im_h, config.im_w, 1), upsampling_rate=config.upsampling_rate)()

optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=0.9)
loss = ['mse']
metrics = [utils.ssim, utils.psnr]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# callbacks -- save model
Path(config.model_dir).mkdir(parents=True, exist_ok=True)
model_path = os.path.join(config.model_dir, 'model.{epoch:02d}-{val_loss:.5f}.h5')
saving = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_freq='epoch',
                                            save_best_only=True,
                                            save_weights_only=True)

# callbacks -- log training
write_freq = int(train_count / config.batch_size / 10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir,
                                             write_graph=True,
                                             update_freq=write_freq)
image_gen_val = callbacks.GenerateImages(model,
                                         validation_dataset,
                                         config.job_dir,
                                         interval=write_freq,
                                         postfix='val')
image_gen = callbacks.GenerateImages(model,
                                     train_dataset,
                                     config.job_dir,
                                     interval=write_freq,
                                     postfix='train')

# callbacks -- start tensorboard
start_tensorboard = callbacks.StartTensorBoard(config.job_dir)


# callbacks -- log code and trained models
log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)



model.fit(train_dataset,
          steps_per_epoch=int(train_count / config.batch_size),
          validation_data=validation_dataset,
          validation_steps=int(validation_count / config.batch_size),
          epochs=config.num_epochs,
          callbacks=[
              saving, tensorboard, start_tensorboard, log_code,
              copy_keras, image_gen, image_gen_val
          ])
