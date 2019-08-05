import tensorflow_datasets as tfds
import tensorflow as tf
from trainer.utils.image import get_central_mask2d

def get_oxford_flowers_dataset(train_type='train', size=(256, 256, 3), mask_size=(128,128), batch_size=32):
    count = {
        'test': 6149,
        'train': 1020,
        'validation': 1020        
    }

    data = tfds.load('oxford_flowers102')
    dataset = data[train_type]
    dataset = dataset.map(lambda x: tf.image.random_crop(x['image'], size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: get_mask_image_pair(x, mask_size=mask_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat().batch(batch_size).prefetch(2)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]
    
def get_mask_image_pair(image, mask_size=(128, 128)):
    mask, bbox = get_central_mask2d(size=mask_size, template_shape=tf.shape(image)[:-1])
    image = normalize(image)
    mask = tf.cast(mask, dtype=tf.float32)
    return image, mask

def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    return 2.0*(image-tf.math.reduce_min(image))/(tf.math.reduce_max(image)-tf.math.reduce_min(image))-1.0