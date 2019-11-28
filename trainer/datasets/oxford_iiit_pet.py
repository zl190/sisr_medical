import tensorflow_datasets as tfds
import tensorflow as tf
from trainer.utils.bicubic_downsample import build_filter, apply_bicubic_downsample

def get_oxford_iiit_pet_dataset(train_type='train', size=(224, 224, 3), downsampling_factor=4, batch_size=32):
    count = {
        'test': 3669,
        'train': 3680,
    }

    data = tfds.load('oxford_iiit_pet')
    dataset = data[train_type]
#     dataset = dataset.map(lambda x: tf.image.resize_with_crop_or_pad(x['image'], target_height=size[0], target_width=size[1]), 
#                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.map(lambda x: tf.cast(x, dtype=tf.float32)/255.0, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if train_type == 'train':
        dataset = dataset.map(lambda x: random_jitter(x['image']), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif train_type == 'test':
        dataset = dataset.map(lambda x: tf.image.resize(x['image'], [size[0], size[1]]), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
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


    

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def random_crop(input_image):
    cropped_image = tf.image.random_crop(
      input_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image



IMG_WIDTH = 224
IMG_HEIGHT = 224

@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    input_image = resize(input_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image = random_crop(input_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)

    return input_image



def load_image_train(img_path, label):
    input_image, label = _load_image(img_path, label)
    input_image = random_jitter(input_image)

    input_image = normalize(input_image)

    return input_image, label

def load_image_test(img_path, label):
    input_image, label = _load_image(img_path, label)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize(input_image)

    return input_image, label