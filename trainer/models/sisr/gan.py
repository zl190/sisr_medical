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
from trainer.models.sisr import MySRResNet, Discriminator
from trainer import utils



#Load VGG model
from tensorflow.keras import models, optimizers, metrics
from tensorflow.keras.applications.vgg19 import preprocess_input
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [224,224,3])
vgg.trainable = False
content_layers = 'block5_conv2'

lossModel = models.Model([vgg.input], vgg.get_layer(content_layers).output, name = 'vggL')

def _lossMSE(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_true - y_pred))

def _lossVGG(y_true, y_pred):
  Xt = preprocess_input(y_pred*255)
  Yt = preprocess_input(y_true*255)
  vggX = lossModel(Xt)
  vggY = lossModel(Yt)
  return tf.reduce_mean(tf.square(vggY-vggX))

def _lossGAN(y_pred):
  """
    params:
    X: hr_pred
  """
  return tf.sum(-1*tf.math.log(D(y_pred)))


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def d_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def g_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
  
  
class MySRGAN:   
    def __init__(self, g=None, d=None,
                 hr_shape=(None, None, 3),
                 lr_shape=(None, None, 3),
                 GAN_LOSS_ALPHA = 0.001,
                 NUM_ITER = 1):
        
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape
        
        if g is None or d is None:
            self.d = Discriminator()
            self.g = MySRResNet()
        else:
            self.d, self.g = d, g
        
        self.GAN_LOSS_ALPHA = GAN_LOSS_ALPHA
        self.NUM_ITER = NUM_ITER
        
    def get_models(self):
        return self.g, self.d
        
    def compile(self, optimizer=None, metrics=[]):
        if optimizer is None: raise Exception('optimizer cannot be None')
    
        self.optimizer = optimizer
        self.metrics = metrics
        
        # Inputs
        lr = tf.keras.layers.Input(shape=self.lr_shape)   # Input images from both domains
        hr = tf.keras.layers.Input(shape=self.hr_shape)

        # Build the critics
        self.g.trainable = False
        self.d.trainable = True
        
        real_image = hr
        fake_image = self.g(lr)
        global_valid = self.d(real_image)
        global_fake = self.d(fake_image)
        
        self.global_critic_model = tf.keras.Model(inputs=[lr, hr],
                                                  outputs=[
                                                      global_valid, 
                                                      global_fake
                                                  ])
        
        self.global_critic_model.compile(optimizer=optimizer,
                                         loss=[
                                              cross_entropy,
                                              cross_entropy
                                         ],
                                         loss_weights=[1,1])
        
        
        
        # Build the Generator
        self.d.trainable = False
        self.g.trainable = True
        fake_image = self.g(lr)
        global_valid = self.d(fake_image)
               
        self.combined_generator_model = tf.keras.Model(inputs=lr, 
                                                       outputs=[
                                                         global_valid, 
                                                         fake_image
                                                       ])

        self.combined_generator_model.compile(optimizer=optimizer, 
                                              loss=[
                                                cross_entropy,
                                                'mse'
                                              ],
                                              loss_weights=[
                                                self.GAN_LOSS_ALPHA, 
                                                1
                                              ])
        
    def validate(self, validation_steps):
        """Returns a dictionary of numpy scalars"""
        metrics_summary = {
            'd_loss': [],
#             'd_val_loss': [],
#             'd_inval_loss': [],
            'g_loss': [],
        }
        
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []
        
        for step in range(validation_steps):
            lr, hr = next(self.dataset_val_next)

            d_global_loss = self.global_critic_model.test_on_batch([lr, hr], 
                                                                    [np.ones((hr.shape[0],1)), # valid
                                                                     np.zeros((hr.shape[0],1)), # invalid
                                                                     ]) 

            g_loss = self.combined_generator_model.test_on_batch([lr, hr],
                                                                 [np.ones((hr.shape[0],1)), # valid global
                                                                  hr]) # mse loss
            
            # Log important metrics
            fake_B = self.g.predict(lr)
            metrics_summary['d_loss'].append(d_global_loss[0]+d_global_loss[1])
#             metrics_summary['d_val_loss'].append(d_global_loss[0])
#             metrics_summary['d_inval_loss'].append(d_global_loss[1])
            metrics_summary['g_loss'].append(g_loss[0]+g_loss[1])
            
            for metric in self.metrics:
                metrics_summary[metric.__name__].append(metric(hr, fake_B).numpy())
                        
        # average all metrics
        for key, value in metrics_summary.items():
            metrics_summary[key] = np.mean(value)
        return metrics_summary
    

    def _fit_init(self, dataset, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose):
        """Initialize Callbacks and Datasets"""
        if not hasattr(self, 'dataset_next'):
            self.dataset_next = iter(dataset)
            metric_names = ['d_loss', 
                            'g_loss', 
                            #'d_val_loss', 
                            #'d_inval_loss'
                           ]
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
                
#                 for i in range(self.NUM_ITER): # Train critics more than generator
#                     lr, hr = next(self.dataset_next)
#                     d_global_loss = self.global_critic_model.train_on_batch([lr, hr], 
#                                                                             [np.ones((hr.shape[0],1)), # valid
#                                                                              np.zeros((hr.shape[0],1)), # invalid
#                                                                              ]) 
                
                lr, hr = next(self.dataset_next)
                d_global_loss = self.global_critic_model.train_on_batch([lr, hr], 
                                                                        [np.ones((hr.shape[0],1)), # valid
                                                                         np.zeros((hr.shape[0],1)), # invalid
                                                                        ])
                g_loss = self.combined_generator_model.train_on_batch([lr, hr],
                                                                      [np.ones((hr.shape[0],1)), # valid 
                                                                       hr, # MSE loss
                                                                      ]) 
                
                # Log important metrics
                fake_image = self.g.predict(lr)
                self.log['g_loss'] = g_loss[0] + g_loss[1]
#                 self.log['d_val_loss'] = d_global_loss[0]
#                 self.log['d_inval_loss'] = d_global_loss[1]
                self.log['d_loss'] = d_global_loss[0]+d_global_loss[1]
                
                for metric in self.metrics:
                    self.log[metric.__name__] = metric(hr, fake_image)
                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
