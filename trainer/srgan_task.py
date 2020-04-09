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
import os

# dataset
train_dataset, train_count = datasets.deeplesion_lr_hr_pair(split='train', 
                                                           size=(config.im_h, config.im_w, 1), 
                                                           downsampling_factor=config.upsampling_rate, 
                                                           batch_size=config.batch_size, 
                                                           augment=False, 
                                                   data_dir = config.data_dir)
validation_dataset, validation_count = datasets.deeplesion_lr_hr_pair(split='validation', 
                                                                     size=(config.im_h, config.im_w, 1), 
                                                                     downsampling_factor=config.upsampling_rate, 
                                                                     batch_size=config.batch_size, 
                                                                     augment=False, 
                                                                     data_dir = config.data_dir)

# Compile or load the model
if config.g_weight == None or config.d_weight == None:
    model = models.sisr.MySRGAN(shape=(config.im_h, config.im_w, 1), 
                                upsampling_rate=config.upsampling_rate,
                                L1_LOSS_ALPHA = config.L1_LOSS_ALPHA,
                                GAN_LOSS_ALPHA = config.GAN_LOSS_ALPHA,
                                )
else:
    d = tf.keras.models.load_model(config.d_weight, compile=False)
    g = tf.keras.models.load_model(config.g_weight, 
                                   custom_objects={"ssim": utils.ssim, "psnr":utils.psnr})

    d._name = 'd1'
    g._name = 'g1'
    model = models.sisr.MySRGAN(g=g, 
                                d=d,
                                shape=(config.im_h, config.im_w, 1), 
                                upsampling_rate=config.upsampling_rate,
                                L1_LOSS_ALPHA = config.L1_LOSS_ALPHA,
                                GAN_LOSS_ALPHA = config.GAN_LOSS_ALPHA,
                                )

    
generator_model, discriminator_model = model.get_models()
model.compile(optimizer=tf.keras.optimizers.Adam(config.lr, beta_1=0.9), metrics=[utils.ssim, utils.psnr])

# Callbacks -- save model
Path(config.model_dir).mkdir(parents=True, exist_ok=True)
model_path = os.path.join(config.model_dir, 'model.{epoch:02d}-{val_g_loss:.5f}.h5')
saving = tf.keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_g_loss', 
                                            verbose=1, 
                                            save_freq='epoch', 
                                            save_best_only=False,
                                            save_weights_only=True)

# callbacks -- log training
write_freq = int(train_count/config.batch_size/10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, 
                                             write_graph=True, 
                                             update_freq=write_freq)
prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')

# callbacks -- start tensorboard
start_tensorboard = callbacks.StartTensorBoard(config.job_dir)


# callbacks -- log code and trained models
log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)




# Fit model
model.fit(train_dataset,
          steps_per_epoch=int(train_count/config.batch_size),
          epochs=config.num_epochs,
          validation_data=validation_dataset,
          validation_steps=int(validation_count/config.batch_size),
          verbose=1,
          callbacks=[
            log_code, 
            prog_bar, 
            start_tensorboard, 
            tensorboard,      
            image_gen, 
            image_gen_val, 
            saving, 
            copy_keras
          ])
