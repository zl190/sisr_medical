#!/usr/bin/env python
# coding: utf-8

# Setup Global Env
env_path = {
  'customed_tfds':'../../datasets/', # tfds package (with deeplesion)
}

import os, sys
for k, v in env_path.items(): 
  if v not in sys.path:
    sys.path.insert(0, v)

import tensorflow as tf
from trainer import utils, models, callbacks, datasets, config
from trainer.models.sisr import MySRResNet, Discriminator
from pathlib import Path

_DATA_DIR = '/datacommons/plusds/team10-zl190/Spring20/tensorflow_datasets' # data_dir


def load_clf_data(split, batch_size, weak_model):
  dataset, cnt = deeplesion_lr_hr_pair(split, size=(config.im_h, config.im_w, 1), batch_size=1, factor=config.upsampling_rate, augment=False, data_dir=config.data_dir)
  dataset = dataset.take(cnt)
  dataset = dataset.unbatch() 
  
  dataset_lr = dataset.map(lambda x, y: x,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset_hr = dataset.map(lambda x, y: y,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset_weak = dataset_lr.map(lambda x: weak_model(tf.expand_dims(x,0), training=False)[0],
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  # label data
  dataset_weak = dataset_weak.map(lambda x: (x, 0)) 
  dataset_hr = dataset_hr.map(lambda x: (x, 1))   

  # interleaves elements from datasets at random
  dataset = tf.data.experimental.sample_from_datasets([dataset_hr, dataset_weak])

  # batch
  dataset = dataset.repeat().batch(batch_size).prefetch(16)
  return dataset, cnt*2


# dataset
weak_model = tf.keras.models.load_model(config.g_weight, compile=False)
train_dataset, train_count = load_clf_data('train', config.batch_size, weak_model)
validation_dataset, validation_count = load_clf_data('validation', config.batch_size, weak_model)


# model
if config.d_weight:
  model = tf.keras.models.load_model(config.d_weight, compile=False)
else:
  model = Discriminator(shape=(config.im_h, config.im_w, 1))()

# optimizer & loss & metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,beta_1=0.9)
loss = [tf.keras.losses.BinaryCrossentropy(name='loss')]
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
          tf.keras.metrics.AUC(name='auc')]

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


history = model.fit(train_dataset,
            steps_per_epoch=int(train_count / config.batch_size),
            validation_data=validation_dataset,
            validation_steps=int(validation_count / config.batch_size),
            epochs=config.num_epochs,
            callbacks=[saving])
