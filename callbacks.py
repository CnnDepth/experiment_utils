from keras.callbacks import LambdaCallback
import numpy as np
import os

class LoggingCallback():
    def __init__(self, model, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.val_losses = []
        self.losses = []
        self.epoch = 0
        self.model = model
    
    def clear_losses(self, epoch, logs):
        self.losses = []
        self.epoch += 1
    
    def write_loss(self, batch, logs):
        self.losses.append(logs['loss'])

    def save_loss_and_model(self, epoch, logs):
        self.model.save(os.path.join(self.save_dir, 'model_on_epoch' + str(self.epoch) + '.hdf5'))
        np.savetxt(os.path.join(self.save_dir, 'losses_on_epoch' + str(self.epoch) + '.txt'),
                   np.array(self.losses)
                  )
        self.val_losses.append(logs['val_loss'])
        np.savetxt(os.path.join(self.save_dir, 'val_losses.txt'), np.array(self.val_losses))
        
    def get_callback(self):
        callback = LambdaCallback(on_epoch_begin=self.clear_losses,
                                  on_epoch_end=self.save_loss_and_model,
                                  on_batch_begin=None,
                                  on_batch_end=self.write_loss
                                 )
        return callback