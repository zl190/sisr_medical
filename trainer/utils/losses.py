import tensorflow as tf


def accuracy(y_true, y_pred): return tf.keras.metrics.Accuracy(y_true, y_pred)
def mse(y_true, y_pred): return tf.keras.losses.MSE(y_true, y_pred)
def mae(y_true, y_pred): return tf.keras.losses.MAE(y_true, y_pred)
def c_mse(y_true, y_pred): 
    y_true_c = tf.image.central_crop(y_true, 0.9)
    y_pred_c = tf.image.central_crop(y_pred, 0.9)
    return tf.keras.losses.MSE(y_true_c, y_pred_c)

def ssim_multiscale(y_true, y_pred): return tf.image.ssim_multiscale(y_true, y_pred, 1)
def ssim(y_true, y_pred): return tf.image.ssim(y_true, y_pred, 1)
def c_ssim(y_true, y_pred): 
    y_true_c = tf.image.central_crop(y_true, 0.9)
    y_pred_c = tf.image.central_crop(y_pred, 0.9)
    return tf.image.ssim(y_true_c, y_pred_c, 1)

def psnr(y_true, y_pred): return tf.image.psnr(y_true, y_pred, 1)
def c_psnr(y_true, y_pred): 
    y_true_c = tf.image.central_crop(y_true, 0.9)
    y_pred_c = tf.image.central_crop(y_pred, 0.9)
    return tf.image.psnr(y_true_c, y_pred_c, 1)

def ssim_loss(y_true, y_pred): return 1-tf.image.ssim(y_true, y_pred, 1)
def combined_loss(l_ssim=0.8, l_mae=0.1, l_mse=0.1):
    def _combined_loss(y_true, y_pred):
        return l_ssim*ssim_loss(y_true, y_pred) + l_mae*tf.abs(y_true - y_pred) +  l_mse*tf.square(y_true - y_pred)
    return _combined_loss

def mask_loss(y_true, y_pred, loss_fn=None, gamma=0.99, **kwargs):
    image, mask = y_true[:,:,:,:-1], y_true[:,:,:,-1:]
    
#     Spatial mask currently assumes crop is always center and same throughout the batch
    mask_shape = tf.shape(mask)
    x, y = tf.meshgrid(tf.range(mask_shape[1]), tf.range(mask_shape[2]))
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    mask_shape = tf.cast(mask_shape, dtype=tf.float32)
    z = tf.math.sqrt(tf.math.square(x - mask_shape[1]/2.0) + tf.math.square(y - mask_shape[2]/2.0))
    spatial_discount = 1.0 - tf.math.pow(gamma, z)
    spatial_discount = spatial_discount[None, :, :, None]
    mask = mask*spatial_discount
    
    predicted_image = y_pred
    y_true = mask*image
    y_pred = mask*y_pred
    return loss_fn(y_true, y_pred)
