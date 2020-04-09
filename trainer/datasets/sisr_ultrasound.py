import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import polarTransform


#@title tensorflow dataset 
def duke_ultrasound_lr_hr_pair(split='train', size=(512, 512), downsampling_factor=4, batch_size=1, data_dir=None):
  """Returns a `tf.data.Dataset` generator, in which each element is 0-1 normalized image 
  """
  count = {
      'A':1362,
      'B':1194,
      'MARK':420,
      'test':438,
      'train':2556,
      'validation':278,
  }

  data = tfds.load('duke_ultrasound', data_dir=data_dir)
  dataset = data[split]

  # reshape dtce and normalizes dtce images from 0 to 1
  dataset = dataset.map(process)
  # Take only dtce and convert images to Cartesian
  dataset = dataset.map(scan_convert_tf_wrapper)
  # dataset = dataset.map(tf_random_rotate_image)
  # resize images
  dataset = dataset.map(lambda x: make_shape_tf_wrapper(x, size))

  # build pair of lr and hr
  dataset = dataset.map(lambda x: get_LR_HR_pair(x, downsampling_factor), 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # batch
  dataset = dataset.repeat().batch(batch_size).prefetch(16)
  return dataset, count[split]


def process(ele):
  """reshape 'dtce' and 0-1 normalizes it"""
  ele['dtce'] = tf.reshape(ele['dtce'], [ele['height'], ele['width']])
  ele['dtce'] = (ele['dtce'] - tf.reduce_min(ele['dtce']))/(tf.reduce_max(ele['dtce']) - tf.reduce_min(ele['dtce']))
  return ele


def scan_convert_tf_wrapper(ele):
  """tf_wrapper for py_function"""
  image = ele['dtce'] 
  irad = ele['initial_radius']
  frad = ele['final_radius']
  iang = ele['initial_angle']
  fang = ele['final_angle']

  image = tuple(tf.py_function(func=scan_convert, 
                        inp= [image, irad, frad, iang, fang],
                        Tout= [tf.float32]))
  image[0].set_shape(None for _ in range(2))
  return image[0]


def make_shape_tf_wrapper(image, shape=None, divisible=16, seed=0):
  """tf_wrapper for py_function"""
  image = tuple(tf.py_function(func=make_shape, inp=[image, shape], Tout=[tf.float32]))
  image[0].set_shape(None for _ in range(3))
  return image[0]


def normalize(image):
  """Returns normalized image (0-1)
  
  Args:
    image: `3D or 4D tf.Tensor`, shape=([None], 512, 512, 1)
  
  Return:
    `tf.Tensor`, shape=(512, 512, 3)
  """
  image = tf.cast(image, dtype=tf.float32)
  return (image-tf.math.reduce_min(image))/(tf.math.reduce_max(image)-tf.math.reduce_min(image))


def get_LR_HR_pair(hr, factor):
  """Returns an image pair
  
  Downsample image along x-axis
  
  Attributes:
    hr: `tensor`, [height, width, channel]
    factor: `int` downsampling factor
    
  Returns:
    a pair of tensor
  """
  lr = hr[:, 0:-1:factor, :]
  lr = normalize(lr)
  hr = normalize(hr)
  return lr, hr


def scan_convert(image, irad, frad, iang, fang):
  """Scan converts beam lines"""
  image, _ = polarTransform.convertToCartesianImage(
      np.transpose(image),
      initialRadius=irad.numpy(),
      finalRadius=frad.numpy(),
      initialAngle=iang.numpy(),
      finalAngle=fang.numpy(),
      hasColor=False,
      order=1)
  image = np.transpose(image[:, int(irad):])
  return image


def make_shape(image, shape=None, divisible=16, seed=0):
  """Will reflection pad or crop to make an image divisible by a number.

  If shape is smaller than the original image, it will be cropped randomly
  If shape is larger than the original image, it will be refection padded
  If shape is None, the image's original shape will be minimally padded to be divisible by a number.

  Arguments:
      image {np.array} -- np.array that is (height, width, channels)

  Keyword Arguments:
      shape {tuple} -- shape of image desired (default: {None})
      seed {number} -- random seed for random cropping (default: {0})
      divisible {number} -- number to be divisible by (default: {16})

  Returns:
      np.array, (int, int) -- divisible image no matter the shape, and a tuple of the original size.
  """
  np.random.seed(seed=seed)
  image_height = image.shape[0]
  image_width = image.shape[1]

  shape = shape if shape is not None else image.shape
  height = shape[0] if shape[0] % divisible == 0 else (divisible - shape[0] % divisible) + shape[0]
  width = shape[1] if shape[1] % divisible == 0 else (divisible - shape[1] % divisible) + shape[1]

  # Pad data to batch height and width with reflections, and randomly crop
  if image_height < height:
      remainder = height - image_height
      if remainder % 2 == 0:
          image = np.pad(image, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
      else:
          remainder = remainder - 1
          image = np.pad(image, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')
  elif image_height > height:
      start = np.random.randint(0, image_height - height)
      image = image[start:start+height, :]

  if image_width < width:
      remainder = width - image_width
      if remainder % 2 == 0:
          image = np.pad(image, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
      else:
          remainder = remainder - 1
          image = np.pad(image, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
  elif image_width > width:
      start = np.random.randint(0, image_width - width)
      image = image[:, start:start+width]
  image = image[:,:, None]
  return image