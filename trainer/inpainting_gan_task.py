import tensorflow as tf
import argparse
from functools import partial
from trainer import utils, models, callbacks, datasets, config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input parser
    parser.add_argument('--bs',       type=int,   help='batch size')
    parser.add_argument('--in_h',     type=int,   help='image input size height')
    parser.add_argument('--in_w',     type=int,   help='image input size width')
    parser.add_argument('--in_lh',    type=int,   help='image local input size height')
    parser.add_argument('--in_lw',    type=int,   help='image local input size width')
    parser.add_argument('--epochs',   type=int,   help='number of epochs')
    parser.add_argument('--m',        type=bool,  help='manual run or hp tuning')
    parser.add_argument('--lr',       type=float, help='learning rate')

    # GAN Params
    parser.add_argument('--WGAN_GP_LAMBDA',  type=float, help='gradient penalty importance')
    parser.add_argument('--COARSE_L1_ALPHA', type=float, help='importance of coarse l1')
    parser.add_argument('--L1_LOSS_ALPHA',   type=float, help='importance of fine l1')
    parser.add_argument('--AE_LOSS_ALPHA',   type=float, help='importance of full reconstruction')
    parser.add_argument('--GAN_LOSS_ALPHA',  type=float, help='importance of GAN loss')
    parser.add_argument('--LOCAL',           type=float, help='importance of local patch')
    parser.add_argument('--NUM_ITER',        type=int,   help='number of discriminator iterations per generator iteration')

    # Cloud ML Params
    parser.add_argument('--job-dir',         help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir',       help='Local model directory')
    parser.add_argument('--train_csv',       help='Train csv')
    parser.add_argument('--test_csv',        help='Test csv')
    parser.add_argument('--validation_csv',  help='Validation csv')
    parser.add_argument('--image_dir',            help='Local image directory')
    args = parser.parse_args()
    
    # Merge params
    for key in vars(args):
        if parser.get_default(key) is not None:
            setattr(config, key, parser.get_default(key))

# Prepare data
train_dataset, train_count = datasets.get_oxford_flowers_dataset('test', batch_size=config.bs, mask_size=(config.in_lh, config.in_lw), size=(config.in_h, config.in_w, 3))
validation_dataset, validation_count = datasets.get_oxford_flowers_dataset('validation', batch_size=config.bs, mask_size=(config.in_lh, config.in_lw), size=(config.in_h, config.in_w, 3))

# Compile model
model = models.inpainting.InPaintingWGAN(shape=(config.in_h, config.in_w, 3), 
                                         local_shape=(config.in_lh, config.in_lw, 3),
                                         WGAN_GP_LAMBDA = config.WGAN_GP_LAMBDA,
                                         COARSE_L1_ALPHA = config.COARSE_L1_ALPHA,
                                         L1_LOSS_ALPHA = config.L1_LOSS_ALPHA,
                                         AE_LOSS_ALPHA = config.AE_LOSS_ALPHA,
                                         GAN_LOSS_ALPHA = config.GAN_LOSS_ALPHA,
                                         LOCAL = config.LOCAL,
                                         NUM_ITER = config.NUM_ITER)

generator_model, global_discriminator_model, local_discriminator_model = model.get_models()
model.compile(optimizer=tf.keras.optimizers.Adam(config.lr, beta_1=0.5, beta_2=0.9), metrics=[utils.mae, utils.psnr])
    
# Callbacks
write_freq = int(train_count/config.bs/10)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, write_graph=True, update_freq=write_freq)
prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
saving = tf.keras.callbacks.ModelCheckpoint(config.model_dir + '/model.{epoch:02d}-{val_g_loss:.5f}.hdf5', monitor='val_g_loss', verbose=1, save_freq='epoch', save_best_only=False)

start_tensorboard = callbacks.StartTensorBoard(config.job_dir)
save_multi_model = callbacks.SaveMultiModel([('g', generator_model), ('dg', global_discriminator_model), ('dl', local_discriminator_model)], config.model_dir)
log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)

image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')

# Fit model
model.fit(train_dataset,
          steps_per_epoch=int(train_count/config.bs),
          epochs=config.epochs,
          validation_data=validation_dataset,
          validation_steps=int(validation_count/config.bs/10),
          verbose=1,
          callbacks=[
            log_code, 
            prog_bar, 
            start_tensorboard, 
            tensorboard,      
            image_gen, 
            image_gen_val, 
            saving, 
            save_multi_model, 
            copy_keras])
