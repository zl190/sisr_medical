import tensorflow as tf


def normalize(image):
  """Returns normalized image (0-1)
  
  Args:
    image: `3D or 4D tf.Tensor`, shape=([None], 512, 512, 1)
  
  Return:
    `tf.Tensor`, shape=(512, 512, 3)
  """
  image = tf.cast(image, dtype=tf.float32)
  return (image-tf.math.reduce_min(image))/(tf.math.reduce_max(image)-tf.math.reduce_min(image))
    

@tf.function()
def random_jitter(image, shape):
  """Returns augmented image
  
  jitter, random flip
  
  Args:
    image: `3D or 4D tf.Tensor`, shape=([None], 512, 512, 1)
    shape: `1D tf.Tensor`, ([None], crop_height, crop_width, 1)
  
  Return:
    `tf.Tensor`, shape=([None], crop_height, crop_width, 1)
  """
  # resize
  new_height, new_width = shape[0], shape[1]
  image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # enlarge
  new_height_float, new_width_float = tf.cast(new_height, tf.float32), tf.cast(new_width, tf.float32)
  ext_height = tf.cast(tf.round(1.1*new_height_float), tf.int32)
  ext_width = tf.cast(tf.round(1.1*new_width_float), tf.int32)
  image = tf.image.resize(image, [ext_height, ext_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # random crop
  image = tf.image.random_crop(image, size=shape)

  # random flip
  if tf.random.uniform(()) > 0.5:
      # random mirroring
      image = tf.image.flip_left_right(image)

  return image