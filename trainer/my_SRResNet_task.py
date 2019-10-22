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
train_dataset, train_count = datasets.get_oxford_iiit_pet_dataset('train', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))
validation_dataset, validation_count = datasets.get_oxford_iiit_pet_dataset('test', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))


# Compile model
generator_model = models.sisr.MySRResNet()
generator_model.compile(optimizer=tf.keras.optimizers.Adam(config.lr), 
                        loss=[utils.mse],
                        loss_weights = [1.0], # Weights for x1_ae, x2_ae, x1_local, x2_local
                        metrics=[utils.ssim, utils.psnr],
                        )

# Callbacks
write_freq = int(train_count/config.bs/10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, write_graph=True, update_freq=write_freq)

saving = tf.keras.callbacks.ModelCheckpoint(config.model_dir + '/model.{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1, save_freq='epoch', save_best_only=False)

log_code = callbacks.LogCode(config.job_dir, './trainer')
copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)

image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')
start_tensorboard = callbacks.StartTensorBoard(config.job_dir)

# Fit model
generator_model.fit(train_dataset,
                    steps_per_epoch=int(train_count/config.bs),
                    epochs=config.epochs,
                    validation_data=validation_dataset,
                    validation_steps=int(validation_count/config.bs),
                    verbose=1,
                    callbacks=[
                      log_code, 
                      start_tensorboard, 
                      tensorboard, 
                      image_gen, 
                      image_gen_val, 
                      saving, 
                      copy_keras
                    ])
