import tensorflow as tf

def mse(y_true, y_pred): return tf.keras.losses.MSE(y_true, y_pred)
def mae(y_true, y_pred): return tf.keras.losses.MAE(y_true, y_pred)

def ssim_multiscale(y_true, y_pred): return tf.image.ssim_multiscale(y_true, y_pred, 1)
def ssim(y_true, y_pred): return tf.image.ssim(y_true, y_pred, 1)
def psnr(y_true, y_pred): return tf.image.psnr(y_true, y_pred, 1)
def ssim_loss(y_true, y_pred): return 1-tf.image.ssim(y_true, y_pred, 1)
def combined_loss(l_ssim=0.8, l_mae=0.1, l_mse=0.1):
    def _combined_loss(y_true, y_pred):
        return l_ssim*ssim_loss(y_true, y_pred) + l_mae*tf.abs(y_true - y_pred) +  l_mse*tf.square(y_true - y_pred)
    return _combined_loss

def mask_loss(y_true, y_pred, loss_fn=None, **kwargs):
    image, mask = y_true[:,:,:,:-1], y_true[:,:,:,-1:]
    predicted_image = y_pred
    y_true = mask*image
    y_pred = mask*y_pred
    return loss_fn(y_true, y_pred)
