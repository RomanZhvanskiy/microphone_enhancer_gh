'''
    the following subroutines are available in this module:

                     train_model_on_data_from_file,
                     save_model,
                     load_model,
                     model_predict
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#
#from IPython.display import Audio #play back the signal (original waveform)
#import  IPython
from keras.callbacks import EarlyStopping
from keras.callbacks import BackupAndRestore
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pickle

######################################################
#          models which can be used
######################################################

def build_autoencoder_10_256 ():
    model = models.Sequential()
    model.add(layers.Dense(10, input_dim=256, activation='relu'))
    model.add(layers.Dense(256,  activation='relu'))
    return model



def build_convolutional_autoencoder_16_32_64_32_16_1():
    conv_ac = models.Sequential()
    conv_ac.add(layers.Reshape([256,1], input_shape=[256]))
    conv_ac.add(layers.Conv1D(16, kernel_size=6, padding="same", input_dim=[256,1], activation='selu'))
    conv_ac.add(layers.MaxPool1D(pool_size=4))
    conv_ac.add(layers.Conv1D(32, kernel_size=6, padding="same", activation='selu'))
    conv_ac.add(layers.MaxPool1D(pool_size=4))
    conv_ac.add(layers.Conv1D(64, kernel_size=6, padding="same", activation='selu'))
    conv_ac.add(layers.MaxPool1D(pool_size=4))

    conv_ac.add(layers.Conv1DTranspose(32, kernel_size=6, strides=4, padding="same", activation='selu'))
    conv_ac.add(layers.Conv1DTranspose(16, kernel_size=6, strides=4, padding="same", activation='selu'))
    conv_ac.add(layers.Conv1DTranspose(1, kernel_size=6, strides=4, padding="same", activation='relu'))
    return conv_ac


######################################################
#          training
######################################################
def train_model  (train_sg           ,
                                    test_sg            ,
                                    degraded_train_sg  ,
                                    degraded_test_sg   ,
                                    model_type         ,
                                    epochs  ):

#create the model
    dictionary_of_models = {
    "microphone_enhancer_gh/autoencoder_10_256" : build_autoencoder_10_256(),
    "microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1" : build_convolutional_autoencoder_16_32_64_32_16_1(),
    }

    if model_type not in dictionary_of_models:
        print(f"model {model_type} is not implelented")
        return

    model = dictionary_of_models[model_type]

    optimizer = Adam()
    model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mse'])

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    br = BackupAndRestore(
        backup_dir="training_backup",
        save_freq="epoch",
        delete_checkpoint=True
    )

    if (epochs < 2): epochs=2 #epochs=1 doesn't work


    history = model.fit( x= np.transpose(degraded_train_sg),
                                  y= np.transpose(train_sg),
                               batch_size=4,
                               validation_data=(np.transpose(degraded_test_sg), np.transpose(test_sg)),
                               epochs=epochs,
                               verbose=0,
                               workers=24,
                               callbacks=[es,br],
                               use_multiprocessing=True)


    return model, history


def save_model (model, history, filepath):
    models.save_model(model, filepath)

    history_filepath = filepath +  "trainHistoryDict"

    with open(history_filepath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def load_model (filepath):
    model = models.load_model(filepath)
    return model

def model_predict (model, bad_audio_sg):
    good_audio_sg_t = model.predict(np.transpose(bad_audio_sg))
    good_audio_sg = np.transpose(good_audio_sg_t)


    return good_audio_sg


#test some of the methods
def main( ):
    '''tester'''
    #train_model
    print ("testing train_model")
    where_to_save_training_data = "/home/romanz/code/RomanZhvanskiy/microphone_enhancer_gh/Data/postprocessed_training_data"
    train_sg           = np.genfromtxt(f"{where_to_save_training_data}/train_sg.sg",          delimiter=',')
    test_sg            = np.genfromtxt(f"{where_to_save_training_data}/test_sg.sg",           delimiter=',')
    degraded_train_sg  = np.genfromtxt(f"{where_to_save_training_data}/degraded_train_sg.sg", delimiter=',')
    degraded_test_sg   = np.genfromtxt(f"{where_to_save_training_data}/degraded_test_sg.sg",  delimiter=',')

    model, history = train_model(\
                            train_sg          = train_sg          ,
                            test_sg           = test_sg           ,
                            degraded_train_sg = degraded_train_sg ,
                            degraded_test_sg  = degraded_test_sg  ,
                            model_type         = "autoencoder_10_256",
                            only_do_one_epoch = 1)
    print ("finished testing train_model")




if __name__ == '__main__':
    main()
