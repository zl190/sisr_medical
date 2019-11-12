import tensorflow as tf

class SaveMultiModel(tf.keras.callbacks.Callback):
    def __init__(self, models, model_dir):
        self.multi_models = models
        self.model_dir = model_dir
        super()
    
    def save_models(self, epoch):
        for name, model in self.multi_models:
            model.save('{}/{}_{}'.format(self.model_dir, name, epoch), save_format='tf')
    
    def on_epoch_end(self, epoch, logs):
        self.save_models(epoch)
