import tensorflow_datasets as tfds
import tensorflow as tf
from trainer.utils.bicubic_downsample import build_filter, apply_bicubic_downsample


def get_oxford_iiit_pet_dataset_for_D(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'test': 3669,
        'train': 3680,
    }

    data = tfds.load('oxford_iiit_pet')
    dataset = data[train_type]
    

    if train_type == 'train':
        dataset = dataset.map(lambda x: random_jitter(x['image'], size), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif train_type == 'test':
        dataset = dataset.map(lambda x: tf.image.resize(x['image'], [size[0], size[1]]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # downsampling
    dataset_lr = dataset.map(lambda x: get_lr(x, downsampling_factor), 
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # build bicubic data with label
    dataset_bicubic = dataset_lr.map(lambda x: (tf.image.resize(x,[size[0], size[1]], method=tf.image.ResizeMethod.BICUBIC), 0))
    # build hr data with label
    dataset = dataset.map(lambda x: (x, 1))   
    
    dataset = dataset.concatenate(dataset_bicubic)
    dataset = dataset.shuffle(buffer_size=count[train_type]*2) 
    
    dataset = dataset.repeat().batch(batch_size).prefetch(8)

    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]


def get_oxford_iiit_pet_dataset(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'test': 3669,
        'train': 3680,
    }

    data = tfds.load('oxford_iiit_pet')
    dataset = data[train_type]

    if train_type == 'train':
        dataset = dataset.map(lambda x: random_jitter(x['image'], size), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif train_type == 'test':
        dataset = dataset.map(lambda x: tf.image.resize(x['image'], [size[0], size[1]]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.repeat().batch(batch_size).prefetch(16)
    dataset = dataset.map(lambda x: get_LR_HR_pair(x, downsampling_factor), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]


def get_coco_dataset(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'validation': 5000, 
        'train': 118287,
    }

    data = tfds.load('coco/2017')
    dataset = data[train_type]

    if train_type == 'train':
        dataset = dataset.map(lambda x: random_jitter(x['image'], size), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif train_type == 'validation':
        dataset = dataset.map(lambda x: tf.image.resize(x['image'], [size[0], size[1]]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    dataset = dataset.repeat().batch(batch_size).prefetch(16)
    dataset = dataset.map(lambda x: get_LR_HR_pair(x, downsampling_factor), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    return dataset, count[train_type]


def get_lr(hr, downsampling_factor=4):
    """Downsample x which is a tensor with shape [N, H, W, 3]

    """
    hr = tf.expand_dims(hr, 0)
    k = build_filter(factor=4)
    lr = apply_bicubic_downsample(hr, filter=k, factor=4)
    lr = normalize(lr)
    lr = tf.squeeze(lr)
    return lr


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


    

def resize(input_image, shape):
    input_image = tf.image.resize(input_image, shape,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def random_crop(input_image, shape):
    cropped_image = tf.image.random_crop(
      input_image, size=shape)

    return cropped_image


@tf.function()
def random_jitter(input_image, shape):
    # resizing to target_shape: e.g 286 x 286 x 3
    input_image = resize(input_image, [shape[0], shape[1]])
    shape_float = tf.cast(tf.shape(input_image), tf.float32)
    ext_height = tf.cast(tf.round(1.1*shape_float[0]), tf.int32)
    ext_width = tf.cast(tf.round(1.1*shape_float[1]), tf.int32)
    # resizing to a little bit bigger
    input_image = resize(input_image, [ext_height, ext_width])
    # randomly cropping to target_shape: e.g 256 x 256 x 3
    input_image = random_crop(input_image, shape)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)

    return input_image

