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
        image, mask = next(self.dataset_iterator)
        x1, x2, x1c, x2c = self.forward.predict([image, mask])
        with self.summary_writer.as_default():
            tf.summary.image('{}/mask_image'.format(self.postfix), mask, step=self.step_count)
            tf.summary.image('{}/real_image'.format(self.postfix), image, step=self.step_count)
            tf.summary.image('{}/fake_image'.format(self.postfix), x2c, step=self.step_count)
            tf.summary.image('{}/delta_image'.format(self.postfix), tf.abs(x2c-image), step=self.step_count)

    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()
            
    def on_train_end(self, logs={}):
        self.generate_images()
