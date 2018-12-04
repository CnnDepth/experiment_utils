from keras.models import load_model
from keras.utils import multi_gpu_model
import os

def train_model(model_file,
                save_dir,
                params_list,
                callback,
                rgbs_train, depths_train,
                rgbs_val, depths_val):
    # initialize model and callback, create save_dir
    fcrn_model = load_model(model_file)
    fcrn_model_gpu = multi_gpu_model(fcrn_model, gpus=2)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # train the model with given params
    for ii, params in enumerate(params_list):        
        # compile model
        trainable_layers = params['trainable_layers']
        for i, is_trainable in enumerate(trainable_layers):
            fcrn_model.layers[i].trainable = is_trainable
        fcrn_model.compile(optimizer=params['optimizer'], loss='mean_squared_error')
        fcrn_model_gpu.compile(optimizer=params['optimizer'], loss='mean_squared_error')
        
        # train model
        batch_size = params['batch_size']
        n_epochs = params['epochs']
        if params['generator'] is None:
            fcrn_model_gpu.fit(rgbs_train, depths_train,
                               batch_size=batch_size,
                               epochs=n_epochs,
                               validation_data=[rgbs_val, depths_val],
                               callbacks=[callback],
                               shuffle='batch'
                               )
        else:
            fcrn_model_gpu.fit_generator(params['generator'],
                                         steps_per_epoch=len(rgbs_train) // batch_size,
                                         epochs=n_epochs,
                                         validation_data=[rgbs_val, depths_val],
                                         callbacks=[callback]
                                         )
        fcrn_model.save(os.path.join(save_dir, 'model_after_stage_' + str(ii) + '.hdf5'))