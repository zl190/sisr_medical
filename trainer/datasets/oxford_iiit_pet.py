import tensorflow_datasets as tfds
import tensorflow as tf
# from ... import trainer
from trainer.utils.bicubic_downsample import build_filter, apply_bicubic_downsample

def get_oxford_iiit_pet_dataset(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'test': 3669,
        'train': 3680,
    }

    data = tfds.load('oxford_iiit_pet')
    dataset = data[train_type]
    dataset = dataset.map(lambda x: tf.image.resize_with_crop_or_pad(x['image'], target_height=size[0], target_width=size[1]), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, dtype=tf.float32)/255.0, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat().batch(batch_size).prefetch(16)
    dataset = dataset.map(lambda x: get_LR_HR_pair(x, downsampling_factor), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]


def get_LR_HR_pair(hr, downsampling_factor=4):
    """Downsample x which is a tensor with shape [N, H, W, 3]

    """
    k = build_filter(factor=4)
    lr = apply_bicubic_downsample(hr, filter=k, factor=4)
    lr = normalize(lr)
    hr = normalize(hr)
    return lr, hr


def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    return (image-tf.math.reduce_min(image))/(tf.math.reduce_max(image)-tf.math.reduce_min(image))