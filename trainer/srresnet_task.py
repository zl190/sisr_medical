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
from trainer import utils, models, callbacks, datasets, config


# Prepare data
train_dataset, train_count = datasets.get_oxford_iiit_pet_dataset('train', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))
validation_dataset, validation_count = datasets.get_oxford_iiit_pet_dataset('test', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))


# Compile or load the model
if config.g_weight ==None:
    generator_model = models.sisr.MySRResNet(shape=(config.in_lh, config.in_lw, 3))()
    generator_model.compile(optimizer=tf.keras.optimizers.Adam(config.lr), 
                            loss=[utils.mse],
                            loss_weights = [1.0],
                            metrics=[utils.c_ssim, utils.c_psnr],
                            )
else:
    generator_model = tf.keras.models.load_model(config.g_weight, 
                               custom_objects={"c_ssim": utils.c_ssim, "c_psnr":utils.c_psnr})


# Callbacks
write_freq = int(train_count/config.bs/10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, write_graph=True, update_freq=write_freq)

saving = tf.keras.callbacks.ModelCheckpoint(config.model_dir + '/g' + '/model.{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1, save_freq='epoch', save_best_only=True)

log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)

image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')
start_tensorboard = callbacks.StartTensorBoard(config.job_dir)

# Fit model
generator_model.fit(train_dataset,
                    steps_per_epoch=int(train_count/config.bs),
                    epochs=config.epochs,
                    validation_data=validation_dataset,
                    validation_steps=int(validation_count/config.bs),
                    verbose=1,
                    callbacks=[
                      log_code, 
                      start_tensorboard, 
                      tensorboard, 
                      image_gen, 
                      image_gen_val, 
                      saving, 
                      copy_keras
                    ])
