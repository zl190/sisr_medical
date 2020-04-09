_PATH = '../../../datasets/'
import os, sys
if _PATH not in sys.path:
    sys.path.insert(0, _PATH)

import tensorflow as tf
import tensorflow_datasets as tfds
from trainer.datasets.utils import normalize, random_jitter
import numpy as np
from skimage.transform import radon,iradon
import glob


def deeplesion(config_name='abnormal',
               split='train',
               size=(512, 512, 1),
               augment=True,
               data_dir=None):
  """Returns a `tf.data.Dataset` generator, in which each element is an (-1024, 200) HU windowed and 0-1 normalized image 
  
  'abnormal':
    1. take out 'image'
    2. convert pixel intensities to the original Hounsfield unit (HU) values
    3. clip pixel values into the range (-1024, 200)
    4. random jitter the image if augment is True else resize the image to 'size'
    5. normalize pixel values to (0,1)
  
  Args:
    config_name: `str`, name of the configure
    split: `str`, ['train' | 'val' | 'test']
    size: `tuple`, (height, width, channels)
    batch_size: `int`
    augment: `Bool`
    data_dir: `str`, optional, where to store the downloaded and serialized data
    
  Returns:
    `tuple`, (tf.data.Dataset, number of examples)
  """
    count = {'train': 42600, 'validation': 9044, 'test': 9278}

    data = tfds.load('deeplesion/{}'.format(config_name), data_dir=data_dir)
    dataset = data[split]

    dataset = dataset.map(lambda x: (x['image']),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (_debias_HU(x)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.clip_by_value(x, -1024, 200)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # augment
    if augment:
        dataset = dataset.map(lambda x:
                              (random_jitter(x, [size[0], size[1], 1])),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: (tf.image.resize(
            x, [size[0], size[1]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda x: (normalize(x)), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)


    return dataset, count[split]
    

def deeplesion_lr_hr_pair(split='train', size=(512, 512, 1), batch_size=8, factor=4, augment=False, data_dir=None):
  """Returns a `tf.data.Dataset` generator, in which each element is an image pair of its low resolution and high resolution representatives
  
  generate radon space down sampled data and save the data into disk if the data has not existed.
  load the data and batch
  
  Args:
    split: `str`, ['train' | 'val' | 'test']
    size: `tuple`, (height, width, channels)
    batch_size: `int`
    factor: `in`, downsampling factor
    augment: `Bool`
    data_dir: `str`, optional, where to store the downloaded and serialized data
    
  Returns:
    `tuple`, (tf.data.Dataset, number of examples)
  """
  count = {'train': 2556, 
           'validation': 278, 
          }
  PATH_PREFIX = os.path.join(data_dir, "custom", 'sisr_deeplesion', split)
  records = sorted(glob.glob(os.path.join(PATH_PREFIX, '*.tfrecord')))

  if not records: 
    # preprocess the data and save to disk
    dataset, cnt = deeplesion('abnormal', split=split, size=size, augment=augment, data_dir=data_dir)
  
    # build pair of lr and hr
    dataset = dataset.map(lambda x: get_pair_wrapper(x, factor, size), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (normalize(x), normalize(y)), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # lr_hr_pair uses only a subset of total dataset
    assert count[split] <= cnt

    # write to tfrecord because the radon function takes too much time
    NUM_SHARDS = int(count[split] // 10)
    dataset_to_write = dataset.map(lambda x, y: tf.concat([x, y], -1),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_to_write = dataset_to_write.take(count[split])
    dataset_to_write = dataset_to_write.map(tf.io.serialize_tensor)
    dataset_to_write = dataset_to_write.enumerate()
    
    def reduce_func(key, dataset):
      filename = tf.strings.as_string(key, width=3, fill='0') + '.tfrecord'
      filename = tf.strings.join([PATH_PREFIX, '/', filename])
      writer = tf.data.experimental.TFRecordWriter(filename)
      writer.write(dataset.map(lambda _, x: x))
      return tf.data.Dataset.from_tensors(filename)

    dataset_to_write = dataset_to_write.apply(tf.data.experimental.group_by_window(
                          lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max))

    for x in dataset_to_write:
      print(x)

  # load data from disk
  dataset = tf.data.TFRecordDataset(records)
  dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.float32))
  dataset = dataset.map(setup_shape_after_load, 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(lambda x: tf.unstack(x, num=2, axis=-1),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(lambda x, y: (tf.expand_dims(x, -1), tf.expand_dims(y, -1)),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # batch
  dataset = dataset.repeat().batch(batch_size).prefetch(16)

  return dataset, count[split]


def setup_shape_after_load(ele):
  """set tensor shape and return"""
  ele.set_shape((512, 512, 2))
  return ele


def get_pair_wrapper(hr, downsampling_factor, size):
  """wrapper of py_function `get_LR_HR_pair`"""
  pair = tf.py_function(func=get_LR_HR_pair, inp=[hr, downsampling_factor], Tout=[tf.float32, tf.float32])
  pair[0].set_shape(size)
  pair[1].set_shape(size)
  return pair


def get_LR_HR_pair(hr, factor):
  """Returns an image pair

  Downsample image in random space with theta axis

  Attributs:
    hr: `tensor`, ground truth image
    factor: `int`, downsampling factor

  Returns:
    an `numpy.array` image pair in cartesian space 
  """
  gt=np.squeeze(hr.numpy())
  theta = np.linspace(0.0, 180.0, 1000, endpoint=False)
  sinogram = radon(gt, theta=theta, circle=False) # A collection of projections at several angles is called a sinogram, which is a linear transform of the original image

  theta_down = theta[0:1000:factor]
  sparse =iradon(sinogram[:,0:1000:factor],theta=theta_down,circle=False) # FBP is used here
  full =iradon(sinogram,theta=theta,circle=False)

  lr = sparse[:,:, None]
  hr = full[:,:, None]
  return lr, hr


def _debias_HU(image):
  """Returns image with HU values
  
  subtract 32768 from the pixel intensities to obtain the original Hounsfield unit (HU) values
  
  Args:
    image: `3D or 4D tf.Tensor`, shape=([None], 512, 512, 1)
  
  Return:
    `tf.Tensor`, shape=(512, 512, 1)
  """
  image = tf.cast(image, dtype=tf.int32)
  image = tf.cast(image - 32768, dtype=tf.int64)
  return image

