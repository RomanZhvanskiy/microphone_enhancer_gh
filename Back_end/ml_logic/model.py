'''
    the following subroutines are available in this module:

                     train_model_on_data_from_file,
                     save_model,
                     load_model,
                     model_predict
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import torchaudio
from IPython.display import Audio #play back the signal (original waveform)
import  IPython
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import BackupAndRestore

def train_model  (train_sg           ,
                                    test_sg            ,
                                    degraded_train_sg  ,
                                    degraded_test_sg   ,
                                    model_type         ,
                                    only_do_one_epoch=0  )  :

#create the model

    if (model_type ==  "autoencoder_10_256"):
        model = build_the_simplest_model_possible()
        optimizer = Adam()
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mse'])
    else:
        print (f"error! model_type = {model_type} not recognized!")

    from tensorflow.keras import callbacks
    from keras.callbacks import EarlyStopping
    from keras.callbacks import BackupAndRestore

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

    if (only_do_one_epoch):
        epochs=1
    else:
        epochs=1000
    history = simplest_model.fit( x= np.transpose(degraded_train_sg),
                                  y= np.transpose(train_sg),
                               batch_size=4,
                               validation_data=(np.transpose(degraded_test_sg), np.transpose(test_sg)),
                               epochs=epochs,
                               verbose=0,
                               workers=24,
                               callbacks=[es,br],
                               use_multiprocessing=True)


    return model, history


def save_model                     (model, history, filepath):
    models.save_model(model, filepath)

def load_model                     (model, filepath):
    models.load_model(model, filepath)

def model_predict (model, bad_audio_sg):
    good_audio_sg_t = model.predict(np.transpose(bad_audio_sg))
    good_audio_sg = np.transpose(good_audio_sg_t)
    return good_audio_sg
