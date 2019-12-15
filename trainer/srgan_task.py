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
if config.g_weight == None or config.d_weight == None:
    model = models.sisr.MySRGAN(hr_shape=(config.in_h, config.in_w, 3), 
                                lr_shape=(config.in_lh, config.in_lw, 3),
                                L1_LOSS_ALPHA = config.L1_LOSS_ALPHA,
                                GAN_LOSS_ALPHA = config.GAN_LOSS_ALPHA,
                                NUM_ITER = config.NUM_ITER)
else:
    d = tf.keras.models.load_model(config.d_weight)
    g = tf.keras.models.load_model(config.g_weight, 
                                   custom_objects={"c_ssim": utils.c_ssim, "c_psnr":utils.c_psnr})

    d._name = 'd1'
    g._name = 'g1'
    model = models.sisr.MySRGAN(g=g, 
                                d=d,
                                hr_shape=(config.in_h, config.in_w, 3), 
                                lr_shape=(config.in_lh, config.in_lw, 3),
                                L1_LOSS_ALPHA = config.L1_LOSS_ALPHA,
                                GAN_LOSS_ALPHA = config.GAN_LOSS_ALPHA,
                                NUM_ITER = config.NUM_ITER)

    
generator_model, discriminator_model = model.get_models()
model.compile(optimizer=tf.keras.optimizers.Adam(config.lr, beta_1=0.9), metrics=[utils.c_ssim, utils.c_psnr])

# Callbacks
write_freq = int(train_count/config.bs/10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, write_graph=True, update_freq=write_freq)
prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
saving = tf.keras.callbacks.ModelCheckpoint(config.model_dir + '/model.{epoch:02d}-{val_g_loss:.5f}.hdf5', monitor='val_g_loss', verbose=1, save_freq='epoch', save_best_only=False)

start_tensorboard = callbacks.StartTensorBoard(config.job_dir)
save_multi_model = callbacks.SaveMultiModel([('g', generator_model), ('d', discriminator_model)], config.model_dir)

log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)

image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')

# Fit model
model.fit(train_dataset,
          steps_per_epoch=int(train_count/config.bs),
          epochs=config.epochs,
          validation_data=validation_dataset,
          validation_steps=int(validation_count/config.bs),
          verbose=1,
          callbacks=[
            log_code, 
            prog_bar, 
            start_tensorboard, 
            tensorboard,      
            image_gen, 
            image_gen_val, 
            saving, 
            save_multi_model, 
            copy_keras
          ])
