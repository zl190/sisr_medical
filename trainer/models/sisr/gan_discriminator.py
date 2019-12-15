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
# from tensorflow.keras import models, optimizers, metrics
# from tensorflow.keras.applications.vgg19 import preprocess_input
# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [224,224,3])
# vgg.trainable = False
# content_layers = 'block5_conv2'

# lossModel = models.Model([vgg.input], vgg.get_layer(content_layers).output, name = 'vggL')

# def _lossMSE(y_true, y_pred):
#   return tf.reduce_mean(tf.square(y_true - y_pred))

# def _lossVGG(y_true, y_pred):
#   Xt = preprocess_input(y_pred*255)
#   Yt = preprocess_input(y_true*255)
#   vggX = lossModel(Xt)
#   vggY = lossModel(Yt)
#   return tf.reduce_mean(tf.square(vggY-vggX))

# def _lossGAN(y_pred):
#   """
#     params:
#     X: hr_pred
#   """
#   return tf.sum(-1*tf.math.log(D(y_pred)))

LAMBDA = 100
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_real_loss(y_true, disc_real_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    return real_loss

def d_generated_loss(y_true, disc_generated_output):
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    return generated_loss

def g_gan_loss(y_true, disc_generated_output):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    return gan_loss

def l1_loss(y_true, gen_output):
    l1_loss = tf.reduce_mean(tf.abs(y_true - gen_output))
    return l1_loss

def g_loss(disc_generated_output, gen_output, target):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss

  
class MyDiscriminator:   
    def __init__(self, g=None, d=None,
                 hr_shape=(None, None, 3),
                 lr_shape=(None, None, 3),
                 L1_LOSS_ALPHA = 100,
                 GAN_LOSS_ALPHA = 0.001,
                 NUM_ITER = 1):
        
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape
        
        if g is None or d is None:
            self.d = Discriminator(self.hr_shape)()
            self.g = MySRResNet(self.lr_shape)()
        else:
            self.d, self.g = d, g
        
        self.GAN_LOSS_ALPHA = GAN_LOSS_ALPHA
        self.L1_LOSS_ALPHA = L1_LOSS_ALPHA
        self.NUM_ITER = NUM_ITER
        
    def get_models(self):
        return self.g, self.d
        
    def compile(self, optimizer=None, metrics=[]):
        if optimizer is None: raise Exception('optimizer cannot be None')
    
        self.optimizer = optimizer
        self.metrics = metrics
        
        # Inputs
        input_image = tf.keras.layers.Input(shape=self.lr_shape)   # Input images from both domains
        target = tf.keras.layers.Input(shape=self.hr_shape)

        # Build the critics
        self.g.trainable = False
        self.d.trainable = True
        
        disc_real_output = self.d(target)
#         gen_output = self.g(input_image)
        gen_output = tf.image.resize(input_image, 
                                     [tf.shape(target)[1], tf.shape(target)[2]],
                                     method=tf.image.ResizeMethod.BICUBIC) 
        disc_generated_output = self.d(gen_output)

        self.discriminator_model = tf.keras.Model(inputs=[input_image, target],
                                                  outputs=[
                                                      disc_real_output, 
                                                      disc_generated_output
                                                  ])
        
        self.discriminator_model.compile(optimizer=optimizer,
                                         loss=[
                                              cross_entropy,
                                              cross_entropy
                                         ],
                                         loss_weights=[
                                             1,
                                             1
                                         ])
        
        
        
#         # Build the Generator
#         self.d.trainable = False
#         self.g.trainable = True
#         gen_output = self.g(input_image)
#         disc_generated_output = self.d(gen_output)
               
#         self.generator_model = tf.keras.Model(inputs=[input_image], 
#                                              outputs=[
#                                                 gen_output, 
#                                                 disc_generated_output
#                                             ])

#         self.generator_model.compile(optimizer=optimizer, 
#                                               loss=[
#                                                 l1_loss,
#                                                 cross_entropy
#                                               ],
#                                               loss_weights=[
#                                                 self.L1_LOSS_ALPHA,
#                                                 1, 
#                                               ])
        
    def validate(self, validation_steps):
        """Returns a dictionary of numpy scalars"""
        metrics_summary = {
            'd_loss': [],
#             'g_loss': [],
#             'l1_loss': [],
#             'g_gan_loss': [],
            'd_real_loss': [],
            'd_fake_loss': [],
        }
        
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []
        
        for step in range(validation_steps):
            input_image, target = next(self.dataset_val_next)

            d_loss = self.discriminator_model.test_on_batch(x=[input_image, target], 
                                                     y=[
                                                        np.ones((target.shape[0],1)), # d_real_loss
                                                        np.zeros((target.shape[0],1)), # d_generated_loss
                                                        ]) 

#             g_loss = self.generator_model.test_on_batch(x=[input_image],
#                                                         y=[
#                                                            target, # l1_loss
#                                                            np.ones((target.shape[0],1)), # g_gan_loss
#                                                            ]) 
            
            # Log important metrics
            fake_B = self.g.predict(input_image)
            metrics_summary['d_loss'].append(d_loss[0]+d_loss[1])
#             metrics_summary['g_loss'].append(g_loss[0]+g_loss[1])
#             metrics_summary['l1_loss'].append(g_loss[0]/self.L1_LOSS_ALPHA)
#             metrics_summary['g_gan_loss'].append(g_loss[1])
            metrics_summary['d_real_loss'].append(d_loss[0])
            metrics_summary['d_fake_loss'].append(d_loss[1])

            
            for metric in self.metrics:
                metrics_summary[metric.__name__].append(metric(target, fake_B).numpy())
                        
        # average all metrics
        for key, value in metrics_summary.items():
            metrics_summary[key] = np.mean(value)
        return metrics_summary
    

    def _fit_init(self, dataset, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose):
        """Initialize Callbacks and Datasets"""
        if not hasattr(self, 'dataset_next'):
            self.dataset_next = iter(dataset)
            metric_names = ['d_loss', 
#                             'g_loss', 
#                             'l1_loss',
#                             'g_gan_loss',
                            'd_real_loss',
                            'd_fake_loss'
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
#                     input_image, target = next(self.dataset_next)
#                     d_loss = self.discriminator_model.train_on_batch(x=[input_image, target], 
#                                                               y=[np.ones((target.shape[0],1)), # valid
#                                                                  np.zeros((target.shape[0],1)), # invalid
#                                                                 ]) 
                
                input_image, target = next(self.dataset_next)
                d_loss = self.discriminator_model.train_on_batch(x=[input_image, target], 
                                                                 y=[np.ones((target.shape[0],1)), # d_real_loss
                                                                    np.zeros((target.shape[0],1)), # d_generated_loss
                                                                    ])
#                 g_loss = self.generator_model.train_on_batch(x=[input_image],
#                                                              y=[target,
#                                                                 np.ones((target.shape[0],1)), # g_gan_loss 
#                                                                 ]) 
                
                # Log important metrics
                fake_image = self.g.predict(input_image)
#                 self.log['g_loss'] = g_loss[0]+g_loss[1]
                self.log['d_loss'] = d_loss[0]+d_loss[1]
#                 self.log['l1_loss'] = g_loss[0]/self.L1_LOSS_ALPHA
#                 self.log['g_gan_loss'] = g_loss[1]
                self.log['d_real_loss'] = d_loss[0]
                self.log['d_fake_loss'] = d_loss[1]
                
                for metric in self.metrics:
                    self.log[metric.__name__] = metric(target, fake_image)
                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
