import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, LeakyReLU, Dense, MaxPooling2D


class InPainting:
    def __init__(self, shape=(None, None, 1), mask_shape=(None, None, 1), base_filters=32):
        self.base_filters = base_filters
        self.shape = shape
        self.mask_shape = mask_shape
        
    def __call__(self):
        original_inputs = tf.keras.layers.Input(shape=self.shape)
        mask = tf.keras.layers.Input(shape=self.mask_shape) # 1 where missing
        mask_s = tf.keras.layers.Lambda(lambda x: resize(x, scale=0.25))(mask) # TODO scale depends on depth
        inputs = tf.keras.layers.Lambda(lambda x: x[1]*tf.math.abs(1.0 - x[0]))((mask, original_inputs)) # mask the input image
        ones = tf.keras.layers.Lambda(lambda x: tf.ones_like(x))(inputs) # for valid pixels
        
        # stage1, coarse network generation
        x = Concatenate(axis=-1)([inputs, ones, mask]) # TODO add latent noise z

        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(2,2), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(4,4), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(8,8), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(16,16), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(self.base_filters//2, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.shape[-1], (3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0), name='x1')(x)
        x_stage1 = x

        # reset image pixels outside of mask and prep for Stage 2
        x_completed_1 = tf.keras.layers.Lambda(lambda x: x[0]*x[1] + x[2]*tf.math.abs(1.0 - x[1]), name='x1c')((x, mask, inputs))
        xnow = Concatenate(axis=-1)([x_completed_1, ones, mask])

        ## conv branch
        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(xnow)
        x = Conv2D(self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        temp = x
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(2,2), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(4,4), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(8,8), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(16,16), padding='same', activation=tf.nn.elu)(x)
        
        x_hallu = x

        ## attention branch
        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(xnow)
        x = Conv2D(self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Lambda(lambda x: contextual_attention(x[0], x[0], masks=x[1], ksize=3, stride=1, rate=2), name='attention')((x, mask_s))
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        pm = x

        ## combine branches and upsample
        x = Concatenate(axis=-1)([x_hallu, pm])

        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(self.base_filters//2, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.shape[-1], (3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0), name='x2')(x)
        x_stage2 = x
        
        x_completed_2 = tf.keras.layers.Lambda(lambda x: x[0]*x[1] + x[2]*tf.math.abs(1.0 - x[1]), name='x2c')((x, mask, inputs))

        return tf.keras.Model([original_inputs, mask], [x_stage1, x_stage2, x_completed_1, x_completed_2])

    
def wgan_local_discriminator(base_filters=64, shape=(None, None, 1)):
    inputs = tf.keras.layers.Input(shape=shape)
    mask = tf.keras.layers.Input(shape=(shape[0], shape[1], 1)) # 1 where missing
    x = Concatenate(axis=-1)([inputs, mask])  # TODO add latent noise z

    x = Conv2D(base_filters, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(base_filters*2, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(base_filters*4, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(base_filters*8, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    return tf.keras.Model([inputs, mask], x)

def wgan_global_discriminator(base_filters=64, shape=(None, None, 1)):
    inputs = tf.keras.layers.Input(shape=shape)
    mask = tf.keras.layers.Input(shape=(shape[0], shape[1], 1)) # 1 where missing
    x = Concatenate(axis=-1)([inputs, mask])  # TODO add latent noise z

    x = Conv2D(base_filters, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(base_filters*2, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(base_filters*4, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(base_filters*4, (5,5), strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    return tf.keras.Model([inputs, mask], x)

def resize(x, scale=2, to_shape=None, align_corners=True, func=tf.compat.v1.image.resize_nearest_neighbor):
    if to_shape is None:
        x_shape = tf.cast(tf.shape(x), dtype=tf.float32)
        new_xs = [tf.cast(x_shape[1]*scale, dtype=tf.int32), tf.cast(x_shape[2]*scale, dtype=tf.int32)]
        return func(x, new_xs, align_corners=align_corners)
    else:
        return func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)

def contextual_attention(f, b, masks=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10.0, fuse=True):
    # get shapes
    raw_fs = tf.shape(f)
    raw_bs = tf.shape(b)

    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.image.extract_patches(b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_bs[0], -1, kernel, kernel, raw_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate)
    b = resize(b, scale=1./rate)
    fs = tf.shape(f)
    bs = tf.shape(b)
    masks = resize(masks, scale=1./rate) if masks is not None else tf.zeros([bs[0], bs[1], bs[2], 1])

    m = tf.image.extract_patches(masks, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [raw_bs[0], -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

    # from t(H*W*C) to w(b*k*k*c*h*w)
    w = tf.image.extract_patches(b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [fs[0], -1, ksize, ksize, fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

    f_groups = tf.expand_dims(f, axis=1) # b*1*h*w*c
    w_groups = w # b*k*k*c*hw
    raw_w_groups = raw_w # b*k*k*c*hw
    mask_groups = m # b*k*k*c*hw

    fuse_weight = tf.reshape(tf.eye(fuse_k), [fuse_k, fuse_k, 1, 1])

    ## convolve
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[1,2,3], keepdims=True), 0.), tf.float32)
    w_normed = w / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(w), axis=[1,2,3], keepdims=True)), 1e-4)
    y = tf.map_fn(lambda x: tf.nn.conv2d(x[0], x[1], strides=[1,1,1,1], padding='SAME'), (f_groups, w_normed), dtype=tf.float32)

    if fuse:
        y = tf.reshape(y, [raw_bs[0], 1, fs[1]*fs[2], bs[1]*bs[2], 1])
        y = tf.map_fn(lambda x: tf.nn.conv2d(x, fuse_weight, strides=[1,1,1,1], padding='SAME'), y, dtype=tf.float32)
        y = tf.reshape(y, [raw_bs[0], 1, fs[1], fs[2], bs[1], bs[2]])
        y = tf.transpose(y, [0, 1, 3, 2, 5, 4])
        y = tf.reshape(y, [raw_bs[0], 1, fs[1]*fs[2], bs[1]*bs[2], 1])
        y = tf.map_fn(lambda x: tf.nn.conv2d(x, fuse_weight, strides=[1,1,1,1], padding='SAME'), y, dtype=tf.float32)
        y = tf.reshape(y, [raw_bs[0], 1, fs[2], fs[1], bs[2], bs[1]])
        y = tf.transpose(y, [0, 1, 3, 2, 5, 4])
    y = tf.reshape(y, [raw_bs[0], 1, fs[1], fs[2], bs[1]*bs[2]])
    y *=  mm  # mask
    y = tf.nn.softmax(y*softmax_scale, 4)
    y *=  mm  # mask
    y = tf.map_fn(lambda x: tf.nn.conv2d_transpose(x[0], x[1], output_shape=tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]), 
                  (y, raw_w_groups), dtype=tf.float32)/4
    y = tf.squeeze(y)
    y = tf.reshape(y, raw_fs)
    return y
