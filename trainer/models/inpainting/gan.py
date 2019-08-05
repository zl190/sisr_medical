# Original work Copyright (c) 2019 Ouwen Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import tensorflow as tf
import numpy as np
from functools import partial
from trainer.models.inpainting import InPainting, wgan_local_discriminator, wgan_global_discriminator
from trainer import utils
from trainer.utils.image.mask import batch_clip_image


def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def random_interpolates(inputs):
    shape = tf.shape(inputs[0])
    alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1])
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples=None, mask=None, norm=1.0):
    gradients = tf.gradients(y_pred, averaged_samples)[0]
    if mask is None: mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))

class InPaintingWGAN:   
    def __init__(self, g=None, dl=None, dg=None, 
                 shape=(None, None, 1), local_shape=(None, None, 1), 
                 WGAN_GP_LAMBDA = 10,
                 COARSE_L1_ALPHA = 1.2,
                 L1_LOSS_ALPHA = 1.2,
                 AE_LOSS_ALPHA = 1.2,
                 GAN_LOSS_ALPHA = 0.001,
                 LOCAL = 1,
                 NUM_ITER = 5):
        self.shape = shape
        self.local_shape = local_shape
        
        if g is None or dl is None or dg is None:
            self.dl = wgan_local_discriminator(shape=self.local_shape)
            self.dg = wgan_global_discriminator(shape=self.shape)
            self.g = InPainting(shape=(None, None, self.shape[-1]))()
        else:
            self.dl, self.dg, self.g = dl, dg, g
        
        self.WGAN_GP_LAMBDA = WGAN_GP_LAMBDA
        self.COARSE_L1_ALPHA = COARSE_L1_ALPHA
        self.L1_LOSS_ALPHA = L1_LOSS_ALPHA
        self.AE_LOSS_ALPHA = AE_LOSS_ALPHA
        self.GAN_LOSS_ALPHA = GAN_LOSS_ALPHA
        self.LOCAL = LOCAL
        self.NUM_ITER = NUM_ITER
        
    def get_models(self):
        return self.g, self.dg, self.dl
        
    def compile(self, optimizer=None, metrics=[]):
        if optimizer is None: raise Exception('optimizer cannot be None')
    
        self.optimizer = optimizer
        self.metrics = metrics
        
        # Inputs
        real_image = tf.keras.layers.Input(shape=self.shape)   # Input images from both domains
        mask = tf.keras.layers.Input(shape=(self.shape[0], self.shape[1], 1))

        # Build the critics
        self.g.trainable = False
        self.dl.trainable = True
        self.dg.trainable = True
        
        _, _, _, fake_image = self.g([real_image, mask])

        # Build global critic
        global_valid = self.dg([real_image, mask])
        global_fake = self.dg([fake_image, mask])
        global_interpolated_img = tf.keras.layers.Lambda(random_interpolates)((real_image, fake_image))
        global_validity_interpolated = self.dg([global_interpolated_img, mask])
        global_partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=global_interpolated_img, mask=mask)
        global_partial_gp_loss.__name__ = 'global_gradient_penalty'
        
        self.global_critic_model = tf.keras.Model(inputs=[real_image, mask],
                                                  outputs=[global_valid, 
                                                           global_fake, 
                                                           global_validity_interpolated])
        
        self.global_critic_model.compile(loss=[wasserstein_loss,
                                               wasserstein_loss,
                                               global_partial_gp_loss],
                                         optimizer=optimizer,
                                         loss_weights=[1, 1, self.WGAN_GP_LAMBDA])
        
        # Build local critic
        local_fake_image, local_mask = tf.keras.layers.Lambda(lambda x: batch_clip_image(x[0], x[1], self.local_shape[0]))((fake_image, mask))
        local_real_image, local_mask = tf.keras.layers.Lambda(lambda x: batch_clip_image(x[0], x[1], self.local_shape[0]))((real_image, mask))
        
        local_valid = self.dl([local_real_image, local_mask])
        local_fake = self.dl([local_fake_image, local_mask])
        local_interpolated_img = tf.keras.layers.Lambda(random_interpolates)((local_real_image, local_fake_image))
        local_validity_interpolated = self.dl([local_interpolated_img, local_mask])
        local_partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=local_interpolated_img, mask=local_mask)
        local_partial_gp_loss.__name__ = 'local_gradient_penalty'
        
        self.local_critic_model = tf.keras.Model(inputs=[real_image, mask],
                                           outputs=[local_valid, 
                                                    local_fake, 
                                                    local_validity_interpolated])
        
        self.local_critic_model.compile(loss=[wasserstein_loss,
                                              wasserstein_loss,
                                              local_partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, self.WGAN_GP_LAMBDA])
        
        # Build the Generator
        self.dl.trainable = False
        self.dg.trainable = False
        self.g.trainable = True
        x1, x2, x1c, x2c = self.g([real_image, mask])
        local_x2c, local_mask = tf.keras.layers.Lambda(lambda x: batch_clip_image(x[0], x[1], self.local_shape[0]))((x2c, mask))
        local_valid = self.dl([local_x2c, local_mask])
        global_valid = self.dg([x2c, mask])
               
        self.combined_generator_model = tf.keras.Model([real_image, mask], 
                                                       [local_valid, 
                                                        global_valid, 
                                                        x1, x2, 
                                                        x1c, x2c])

        mask_mae = partial(utils.mask_loss, loss_fn=utils.mae)
        mask_mae.__name__ = 'mask_mae'
        self.combined_generator_model.compile(optimizer=optimizer, 
                                              loss=[wasserstein_loss, 
                                                    wasserstein_loss, 
                                                    'mae', 'mae',
                                                    mask_mae, mask_mae],
                                              loss_weights=[self.GAN_LOSS_ALPHA*self.LOCAL, 
                                                            self.GAN_LOSS_ALPHA,
                                                            self.COARSE_L1_ALPHA*self.AE_LOSS_ALPHA,
                                                            self.AE_LOSS_ALPHA,
                                                            self.COARSE_L1_ALPHA*self.L1_LOSS_ALPHA*self.LOCAL,
                                                            self.L1_LOSS_ALPHA*self.LOCAL])
        
    def validate(self, validation_steps):
        """Returns a dictionary of numpy scalars"""
        metrics_summary = {
            'd_loss': [],
            'g_loss': [],
            'd_local': [],
            'd_global': [],
            'g_local': [],
            'g_global': [],
            'gp': []
        }
        
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []
        
        for step in range(validation_steps):
            image, mask = next(self.dataset_val_next)
            image_mask = np.concatenate((image, mask), axis=-1)
            
            d_local_loss = self.local_critic_model.test_on_batch([image, mask], 
                                                                 [np.ones((image.shape[0],1)), # valid
                                                                  -1*np.ones((image.shape[0],1)), # invalid
                                                                  np.zeros((image.shape[0],1))]) # dummy for gradient penalty

            d_global_loss = self.global_critic_model.test_on_batch([image, mask], 
                                                                    [np.ones((image.shape[0],1)), # valid
                                                                     -1*np.ones((image.shape[0],1)), # invalid
                                                                     np.zeros((image.shape[0],1))]) # dummy for gradient penalty

            g_loss = self.combined_generator_model.test_on_batch([image, mask],
                                                                 [np.ones((image.shape[0],1)), # valid local
                                                                  np.ones((image.shape[0],1)), # valid global
                                                                  image, image, image_mask, image_mask]) # x1, x2, x1c, x2c MAE loss
            
            # Log important metrics
            fake_B = self.g.predict([image, mask])
            metrics_summary['d_local'].append(0.5*(d_local_loss[1]+d_local_loss[2]))
            metrics_summary['d_global'].append(0.5*(d_global_loss[1]+d_global_loss[2]))
            metrics_summary['gp'].append(0.5*(d_local_loss[3]+d_global_loss[3]))
            
            metrics_summary['g_local'].append(self.GAN_LOSS_ALPHA*self.LOCAL*g_loss[1])
            metrics_summary['g_global'].append(self.GAN_LOSS_ALPHA*g_loss[2])
            
            metrics_summary['g_loss'].append(g_loss[0])
            metrics_summary['d_loss'].append(0.5*(d_local_loss[0]+d_global_loss[0]))       
            
            for metric in self.metrics:
                metrics_summary[metric.__name__].append(metric(image, fake_B).numpy())
                        
        # average all metrics
        for key, value in metrics_summary.items():
            metrics_summary[key] = np.mean(value)
        return metrics_summary
    

    def _fit_init(self, dataset, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose):
        """Initialize Callbacks and Datasets"""
        if not hasattr(self, 'dataset_next'):
            self.dataset_next = iter(dataset)
            metric_names = ['d_local', 'd_global', 'g_local', 'g_global', 'd_loss', 'g_loss', 'gp']
            metric_names.extend([metric.__name__ for metric in self.metrics])

        if not hasattr(self, 'dataset_val_next') and validation_data is not None:
            self.dataset_val_next = iter(validation_data)
            metric_names.extend(['val_' + name for name in metric_names])

        for callback in callbacks: 
            callback.set_model(self.g) # only set callbacks to the forward generator
            callback.set_params({
                'verbose': verbose,
                'epochs': epochs,
                'steps': steps_per_epoch,
                'metrics': metric_names # for tensorboard callback to know which metrics to log
            })
                
        self.log = {
            'size': batch_size
        }

        
    def fit(self, dataset, batch_size=8, steps_per_epoch=10, epochs=3, validation_data=None, validation_steps=10, verbose=0, callbacks=[]):
        self._fit_init(dataset, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose)
        
        for callback in callbacks: callback.on_train_begin(logs=self.log)
        for epoch in range(epochs):
            for callback in callbacks: callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
                
                for i in range(self.NUM_ITER): # Train critics more than generator
                    image, mask = next(self.dataset_next)
                    image_mask = np.concatenate((image, mask), axis=-1)
                    d_local_loss = self.local_critic_model.train_on_batch([image, mask], 
                                                                          [np.ones((image.shape[0],1)), # valid
                                                                           -1*np.ones((image.shape[0],1)), # invalid
                                                                           np.zeros((image.shape[0],1))]) # dummy for gradient penalty

                    d_global_loss = self.global_critic_model.train_on_batch([image, mask], 
                                                                            [np.ones((image.shape[0],1)), # valid
                                                                             -1*np.ones((image.shape[0],1)), # invalid
                                                                             np.zeros((image.shape[0],1))]) # dummy for gradient penalty
                
                image, mask = next(self.dataset_next)
                image_mask = np.concatenate((image, mask), axis=-1)
                g_loss = self.combined_generator_model.train_on_batch([image, mask],
                                                                      [np.ones((image.shape[0],1)), # valid local
                                                                       np.ones((image.shape[0],1)), # valid global
                                                                       image, image, image_mask, image_mask]) # x1, x2, x1c, x2c MAE loss
                
                # Log important metrics
                x1, x2, x1c, x2c = self.g.predict([image, mask])
                self.log['d_local'] = 0.5*(d_local_loss[1]+d_local_loss[2])
                self.log['d_global'] = 0.5*(d_global_loss[1]+d_global_loss[2])
                self.log['gp'] = 0.5*(d_local_loss[3]+d_global_loss[3])
                
                self.log['g_local'] = self.GAN_LOSS_ALPHA*self.LOCAL*g_loss[1]
                self.log['g_global'] = self.GAN_LOSS_ALPHA*g_loss[2]
                self.log['g_loss'] = g_loss[0]
                self.log['d_loss'] = 0.5*(d_local_loss[0]+d_global_loss[0])
                
                for metric in self.metrics:
                    self.log[metric.__name__] = metric(image, x2c)
                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
