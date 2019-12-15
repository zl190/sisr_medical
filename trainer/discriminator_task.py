import tensorflow as tf
from trainer import utils, models, callbacks, datasets, config


# Prepare data
train_dataset, train_count = datasets.get_oxford_iiit_pet_dataset_for_D('train', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))
validation_dataset, validation_count = datasets.get_oxford_iiit_pet_dataset_for_D('test', batch_size=config.bs, downsampling_factor=4, size=(config.in_h, config.in_w, 3))


# Compile or load the model
if config.d_weight == None:
    d_model = models.sisr.Discriminator(shape=(config.in_h, config.in_w, 3))()

    d_model.compile(optimizer=tf.keras.optimizers.Adam(config.lr2), 
                            loss=[tf.keras.losses.binary_crossentropy],
                            metrics=[tf.keras.metrics.Accuracy()],
                            )
else:
    d_model = tf.keras.models.load_model(config.d_weight)

# Callbacks
write_freq = int(train_count/config.bs/10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.job_dir, write_graph=True, update_freq=write_freq)

saving = tf.keras.callbacks.ModelCheckpoint(config.model_dir + '/d_c_model.{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1, save_freq='epoch', save_best_only=False)

log_code = callbacks.LogCode(config.job_dir, './trainer')
#copy_keras = callbacks.CopyKerasModel(config.model_dir, config.job_dir)

#image_gen_val = callbacks.GenerateImages(generator_model, validation_dataset, config.job_dir, interval=write_freq, postfix='val')
#image_gen = callbacks.GenerateImages(generator_model, train_dataset, config.job_dir, interval=write_freq, postfix='train')
start_tensorboard = callbacks.StartTensorBoard(config.job_dir)

# Fit model
d_model.fit(train_dataset,
                    steps_per_epoch=int(train_count/config.bs),
                    epochs=config.epochs,
                    validation_data=validation_dataset,
                    validation_steps=int(validation_count/config.bs),
                    verbose=1,
                    callbacks=[
                      log_code, 
                      start_tensorboard, 
                      tensorboard, 
                      #image_gen, 
                      #image_gen_val, 
                      saving, 
                      #copy_keras
                    ])
