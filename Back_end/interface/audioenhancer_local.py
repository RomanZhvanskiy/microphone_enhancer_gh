'''
    the following subroutines are available in this module:
    #preprocess data and train model
    def preprocess_and_train(      ) -> None:
    #preprocess data
    def preprocess(                ) -> None:

    #train model
    def train(                     ) -> None:

    #make model provide predictions
    def pred(                      ) -> None:
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


#############################################################
#                     FOLDER STRUCTURE
#############################################################
#
#FOLDER STRUCTURE
#=======================
#├── Back_end
#│   ├── api
#│   │   ├── api_func.py
#│   │   ├── enhancer_api.py
#│   ├── audio_preprocessing
#│   │   ├── preprocessing.py #conversion of waveforms to/from SG, degrade quality, etc
#│   ├── hugging_models
#│   │   ├── hugrestore.py
#│   ├── image_metrics
#│   │   ├── img_metrics.py #image quality metrics
#│   ├── interface
#│   │   ├── audioenhancer_local.py #this is where the functions are to be called by API
#│   ├── ml_logic
#│   │   ├── model.py #NN model is here
#│   ├── params.py #parameters (mostly folder names) here
#│
#├── Data
#│   ├── audio_data
#│   │   ├── audio_in #this is where bad quality input comes in
#│   │   └── audio_out #this is where good quality output comes out
#│   ├── postprocessed_training_data #preprocessed data for training model
#│   │   ├── degraded_test_sg.sg
#│   │   ├── degraded_train_sg.sg
#│   │   ├── test_sg.sg
#│   │   └── train_sg.sg
#│   ├── pretrained_models #where we save models
#│   └── raw_data #data for training model
#│   └── VCTK-Corpus
#├── Front_end
#│   └── test_api.py #web site
#├── jupyter_books
#│   ├── Copy of Speech_enhancement_6-checkpoint.ipynb
#│   └── hugging_conversions_test.ipynb
#├── Makefile
#├── README.md
#├── requirements.txt
#├── setup.py
#├── test_hugging_models.py
#└── Untitled.ipynb
#
#
# Adds higher directory to python modules path
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


sys.path.append("..") # Adds higher directory to python modules path.

print (f"sys.path = {sys.path}")

from audio_preprocessing import preprocessing   as pp              #ok
from image_metrics import img_metrics           as im              #ok
from hugging_models import hugrestore           as hr              #ok

#commented out below for debugging
from ml_logic.model  import train_model, \
                     save_model,                    \
                     load_model,                    \
                     model_predict




from params import *


def preprocess(num_spectrograms=10, num_speaker =0, debug=0) -> None:
    if(debug):
        print ("preprocess is running in debug mode - do nothing")
        pass

        # Create (X_train, y_train, X_val, y_val) without data leaks
        # Create (X_train_processed, X_val_processed) using `preprocessor.py`
        # Save and append the processed chunk to a local CSV at "data_processed_path"

    if (TRAINING_DATA_LOCATION == "use_local_data"):
        where_to_get_training_data = os.path.dirname(os.path.realpath(__file__))
        where_to_get_training_data = where_to_get_training_data.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{TRAINING_DATA_SUBFOLDER}")


        large_data, sr = pp.get_all_speech_as_one_mel(num_spectrograms, num_speaker, where_to_get_training_data, debug = 1)
        train_sg, test_sg = pp.split_spectrogram_in_train_and_test(large_data,0.2, debug=1)
        degraded_train_sg = pp.degrade_quaity(train_sg, sr )
        degraded_test_sg =  pp.degrade_quaity(test_sg, sr )

        where_to_save_training_data = os.path.dirname(os.path.realpath(__file__))
        where_to_save_training_data = where_to_save_training_data.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{POSTPROCESSED_TRAINING_DATA_SUBFOLDER}")

#/home/romanz/code/RomanZhvanskiy/microphone_enhancer_gh/Back_end/audio_preprocessing/wav48


        np.savetxt(fname=f"{where_to_save_training_data}/train_sg.sg", X=train_sg                  , delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/test_sg.sg", X=test_sg                    , delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/degraded_train_sg.sg", X=degraded_train_sg, delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/degraded_test_sg.sg", X=degraded_test_sg  , delimiter = ",")

    print(f"preprocess data is saved in folder  {where_to_save_training_data}")



def train(debug=0, only_do_one_epoch=0) -> None:
    if(debug):
        print ("train is running in debug mode - do nothing")
        pass

    if (TRAINING_DATA_LOCATION == "use_local_data"):

        #load the training data in the memory
        where_to_save_training_data = os.path.dirname(os.path.realpath(__file__))
        where_to_save_training_data = where_to_save_training_data.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{POSTPROCESSED_TRAINING_DATA_SUBFOLDER}")

        train_sg           = np.genfromtxt(f"{where_to_save_training_data}/train_sg.sg",          delimiter=',')
        test_sg            = np.genfromtxt(f"{where_to_save_training_data}/test_sg.sg",           delimiter=',')
        degraded_train_sg  = np.genfromtxt(f"{where_to_save_training_data}/degraded_train_sg.sg", delimiter=',')
        degraded_test_sg   = np.genfromtxt(f"{where_to_save_training_data}/degraded_test_sg.sg",  delimiter=',')

        #train the model


        model, history = train_model(
                            train_sg          = train_sg          ,
                            test_sg           = test_sg           ,
                            degraded_train_sg = degraded_train_sg ,
                            degraded_test_sg  = degraded_test_sg  ,
                            model_type         = MODEL_TYPE,
                            only_do_one_epoch = only_do_one_epoch)

        #save the model to HDD
        where_to_save_trained_model = os.path.dirname(os.path.realpath(__file__))
        where_to_save_trained_model = where_to_save_trained_model.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{MODEL_SUBFOLDER}")

        save_model(model, history, where_to_save_trained_model)

        # print the last value of the metric
        #print (f"history.history.keys() = {history.history.keys()}")
        display_loss = np.min(history.history['loss'])
        display_mse = np.min(history.history['mse'])
        display_val_loss = np.min(history.history['val_loss'])
        display_val_mse = np.min(history.history['val_mse'])


        if (only_do_one_epoch):
            print (f"model has been successfully trained for one epoch. Achieved validation loss = {display_val_loss}")
        else:
            print (f"model has been successfully trained. Achieved validation loss = {display_val_loss}")



def preprocess_and_train(debug=0) -> None:

    preprocess(debug=debug)
    train(debug=debug)

def pred(test_string="none", debug=0) -> None:
    if(debug):
        print ("pred is running in debug mode - change the " +\
               "value of string test_string and do nothing else")
        test_string = "pred has run successfully in debug mode"
        pass

    #load the model from HDD


    where_to_save_trained_model = os.path.dirname(os.path.realpath(__file__))
    where_to_save_trained_model = where_to_save_trained_model.replace(\
                   "/microphone_enhancer_gh/Back_end/interface",     \
                   f"/microphone_enhancer_gh/{MODEL_SUBFOLDER}")

    if not os.path.exists(where_to_save_trained_model):
        print(f'The file {where_to_save_trained_model} does not exist. Please train the model before making predictions.')
        return


    model = load_model(where_to_save_trained_model)

    #load the bad quality SG from HDD


    where_to_find_bad_audio = os.path.dirname(os.path.realpath(__file__))
    where_to_find_bad_audio = where_to_find_bad_audio.replace(\
                   "/microphone_enhancer_gh/Back_end/interface",     \
                   f"/microphone_enhancer_gh/{BAD_QUALITY_FILE}")

    if not os.path.exists(where_to_find_bad_audio):
        print(f'The file {where_to_find_bad_audio} does not exist. Please provide input to make predictions.')
        return



    bad_audio, sr = pp.load_wav(where_to_find_bad_audio)
    #convert bad_audio to spectrogram

    bad_audio_sg = pp.waveform_2_spectrogram (bad_audio , sr)

    #make predictions
    good_quality_sg = model_predict(
        model = model             ,
        bad_audio_sg = bad_audio_sg )

    #convert good_quality_sg to waveform

    good_audio = pp.spectrogram_2_waveform (good_quality_sg , sr)


    #write the good quality waveform to HDD
    where_to_find_good_audio = os.path.dirname(os.path.realpath(__file__))
    where_to_find_good_audio = where_to_find_good_audio.replace(\
                   "/microphone_enhancer_gh/Back_end/interface",     \
                   f"/microphone_enhancer_gh/{GOOD_QUALITY_FILE}")

    pp.waveform_2_file (good_audio,sr,filename=where_to_find_good_audio)


#test some of the methods
def main( ):
    '''tester'''
    print (f"testing preprocess method")
    preprocess(num_spectrograms=2, num_speaker =0, debug=0)


if __name__ == '__main__':
    main()
