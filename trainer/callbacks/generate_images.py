import tensorflow as tf
import argparse
import numpy as np
from trainer import config

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, dataset, log_dir, interval=1000, postfix='val'):
        super()
        self.step_count = 0
        self.postfix = postfix
        self.interval = interval
        self.forward = forward
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.dataset_iterator = iter(dataset)
        
    def generate_images(self):
        lr, hr = next(self.dataset_iterator)
        hr_pred = self.forward.predict(lr)
        with self.summary_writer.as_default():
            tf.summary.image('{}/lr_image'.format(self.postfix), lr, step=self.step_count)
            tf.summary.image('{}/bicubic_image'.format(self.postfix), 
                             tf.image.resize(lr, 
                                             [tf.shape(hr)[0], tf.shape(hr)[1]], 
                                             method=tf.image.ResizeMethod.BICUBIC), 
                             step=self.step_count)

            tf.summary.image('{}/sr_image'.format(self.postfix), hr_pred, step=self.step_count)
            tf.summary.image('{}/original_image'.format(self.postfix), hr, step=self.step_count)

    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()
            
    def on_train_end(self, logs={}):
        self.generate_images()
