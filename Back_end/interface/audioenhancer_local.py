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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import torchaudio
from IPython.display import Audio #play back the signal (original waveform)
import  IPython
#from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import BackupAndRestore


import shutil #for copying files
from hugging_models.hugrestore import save_spectro_image #for saving pics


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
                     save_model,         \
                     load_model,         \
                     model_predict




from params import *


#############################################################
#            subroutines to provide directory names
#############################################################
def find_a_place_for_trained_model(enhancer="microphone_enhancer_gh/autoencoder_10_256"):
    folder_name = os.path.dirname(os.path.realpath(__file__))
    folder_name = folder_name.replace(\
                   "/microphone_enhancer_gh/Back_end/interface",     \
                   f"/microphone_enhancer_gh/{MODEL_SUBFOLDER}/{enhancer}")
    folder_name = folder_name.replace("/freshly_trained_model/", "/")#remove 1 directlry layer

    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name

def find_a_place_for_training_data ():
    folder_name = os.path.dirname(os.path.realpath(__file__))
    folder_name = folder_name.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{TRAINING_DATA_SUBFOLDER}")
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name

def find_a_place_for_postprocessed_training_data ():
    folder_name = os.path.dirname(os.path.realpath(__file__))
    folder_name = folder_name.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{POSTPROCESSED_TRAINING_DATA_SUBFOLDER}")

    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name

def find_a_place_for_bad_audio ():
    folder_name = os.path.dirname(os.path.realpath(__file__))
    folder_name = folder_name.replace(\
                       "/microphone_enhancer_gh/Back_end/interface",     \
                       f"/microphone_enhancer_gh/{BAD_QUALITY_FILE}")
    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name

def find_a_place_for_good_audio ():
    folder_name = os.path.dirname(os.path.realpath(__file__))
    folder_name = folder_name.replace(\
                   "/microphone_enhancer_gh/Back_end/interface",     \
                   f"/microphone_enhancer_gh/{GOOD_QUALITY_FILE}")

    folder_name = folder_name.replace("//", "/").replace("//", "/")
    return folder_name


#############################################################
#            preprocess
#############################################################

def preprocess(num_spectrograms=10, num_speaker =0, debug=0) -> None:
    if(debug):
        print ("preprocess is running in debug mode - do nothing")
        pass

        # Create (X_train, y_train, X_val, y_val) without data leaks
        # Create (X_train_processed, X_val_processed) using `preprocessor.py`
        # Save and append the processed chunk to a local CSV at "data_processed_path"

    if (TRAINING_DATA_LOCATION == "use_local_data"):
        where_to_get_training_data = find_a_place_for_training_data()


        large_data, sr = pp.get_all_speech_as_one_mel(num_spectrograms, num_speaker, where_to_get_training_data, debug = 1)
        train_sg, test_sg = pp.split_spectrogram_in_train_and_test(large_data,0.2, debug=1)
        degraded_train_sg = pp.degrade_quaity(train_sg, sr )
        degraded_test_sg =  pp.degrade_quaity(test_sg, sr )

        where_to_save_training_data = find_a_place_for_postprocessed_training_data()

#/home/romanz/code/RomanZhvanskiy/microphone_enhancer_gh/Back_end/audio_preprocessing/wav48

        np.savetxt(fname=f"{where_to_save_training_data}/train_sg.sg", X=train_sg                  , delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/test_sg.sg", X=test_sg                    , delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/degraded_train_sg.sg", X=degraded_train_sg, delimiter = ",")
        np.savetxt(fname=f"{where_to_save_training_data}/degraded_test_sg.sg", X=degraded_test_sg  , delimiter = ",")

    print(f"preprocess data is saved in folder  {where_to_save_training_data}")

#############################################################
#            train
#############################################################
def train(debug=0,
          only_do_one_epoch=0, #deprecated
          epochs=1000,
          enhancer=DEFAULT_MODEL_TYPE) -> None:
    if(debug):
        print ("train is running in debug mode - do nothing")
        pass

    if (TRAINING_DATA_LOCATION == "use_local_data"):
        if (only_do_one_epoch):
            epochs = 2 # 1 epocch doesn't work

        #load the training data in the memory
        where_to_save_training_data = find_a_place_for_postprocessed_training_data ()
        print (f"where_to_save_training_data = {where_to_save_training_data}")
        train_sg           = np.genfromtxt(f"{where_to_save_training_data}/train_sg.sg",          delimiter=',')
        test_sg            = np.genfromtxt(f"{where_to_save_training_data}/test_sg.sg",           delimiter=',')
        degraded_train_sg  = np.genfromtxt(f"{where_to_save_training_data}/degraded_train_sg.sg", delimiter=',')
        degraded_test_sg   = np.genfromtxt(f"{where_to_save_training_data}/degraded_test_sg.sg",  delimiter=',')


        if (len(train_sg.shape          ) == 1 or \
			len(test_sg.shape           ) == 1 or  \
			len(degraded_train_sg.shape ) == 1 or\
			len(degraded_test_sg.shape  ) == 1 ) :
            print ("There is a problem with preprocessed training data." +\
                  " Please run make run_preprocess before make run_train.")
            return




        #print (f"train_sg.shape          = {train_sg.shape         }")
        #print (f"test_sg.shape           = {test_sg.shape          }")
        #print (f"degraded_train_sg.shape = {degraded_train_sg.shape}")
        #print (f"degraded_test_sg.shape  = {degraded_test_sg.shape }")
#


        #train the model
        model, history = train_model(
                            train_sg          = train_sg          ,
                            test_sg           = test_sg           ,
                            degraded_train_sg = degraded_train_sg ,
                            degraded_test_sg  = degraded_test_sg  ,
                            model_type         = enhancer,
                            epochs = epochs)

        #save the model to HDD

        where_to_save_trained_model = find_a_place_for_trained_model (enhancer=enhancer)
        #print (f"where_to_save_trained_model = {where_to_save_trained_model}")

        save_model(model, history, where_to_save_trained_model)

        # print the last value of the metric
        #print (f"history.history.keys() = {history.history.keys()}")
        display_loss = np.min(history.history['loss'])
        display_mse = np.min(history.history['mse'])
        display_val_loss = np.min(history.history['val_loss'])
        display_val_mse = np.min(history.history['val_mse'])


        print (f"model has been successfully trained for {epochs} epochs. Achieved validation loss = {display_val_loss}")



def preprocess_and_train(enhancer=DEFAULT_MODEL_TYPE, debug=0) -> None:

    preprocess(debug=debug)
    train(enhancer=enhancer, debug=debug)
#############################################################
#            pred
#############################################################
def pred(test_string="none",
         sr_string="none",
         where_to_find_bad_audio = "not specified",
         debug=0,
         enhancer=DEFAULT_MODEL_TYPE) -> None:
    if(debug):
        print ("pred is running in debug mode - change the " +\
               "value of string test_string and do nothing else")
        test_string = "pred has run successfully in debug mode"
        pass

    #load the model from HDD


    where_to_save_trained_model = find_a_place_for_trained_model(enhancer=enhancer)

    if not os.path.exists(where_to_save_trained_model):
        print(f'The file {where_to_save_trained_model} does not exist.' + \
            ' Please train the model before making predictions.')
        return


    model = load_model(where_to_save_trained_model)

    #load the bad quality SG from HDD

    if(where_to_find_bad_audio == "not specified"):
        where_to_find_bad_audio = find_a_place_for_bad_audio()

    if not os.path.exists(where_to_find_bad_audio):
        print(f'The file {where_to_find_bad_audio} does not exist. ' + \
            ' Please provide input to make predictions.')
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

    #need the following shape: waveform1.shape = (177152,)
    #some models return shape: waveform1.shape =  (1, 177152)
    #rectify this:

    if (len(good_audio.shape) == 2):
        good_audio = good_audio[0, :]


    #write the good quality waveform to HDD
    where_to_find_good_audio = find_a_place_for_good_audio()

    pp.waveform_2_file (waveform1=good_audio,sr=sr,filename=where_to_find_good_audio)



    sr_string = str(sr)

#############################################################
#            pred_for_api
#############################################################
def pred_for_api(where_to_find_bad_audio="not specified",
                 enhancer=DEFAULT_MODEL_TYPE):


    debug = 1
    if (debug):
        with open("debug_dump.txt", "w") as file1:
            # Writing data to a file
            file1.write("pred_for_api reporting \n")
            file1.write(f"where_to_find_bad_audio = {where_to_find_bad_audio}  \n")
            file1.write(f"enhancer = {enhancer}  \n")

    #wrapper for pred, adds spec_and_sr to the output
    # for consistency with other callable models

    #make predictions

    shutil.copyfile(os.path.join(where_to_find_bad_audio), os.path.join(audio_in_path, os.path.basename(f'{GOOD_QUALITY_FILE}'))) # copying temp file to audio_in. file is named '{GOOD_QUALITY_FILE}' for api to work (it is bad_quality actually, distinguish by file location, not name, please)
    save_spectro_image(os.path.join(audio_in_path, os.path.basename(f'{GOOD_QUALITY_FILE}')))

    sr_string = ""


    pred(test_string="none", sr_string=sr_string, enhancer=enhancer, where_to_find_bad_audio = where_to_find_bad_audio, debug=0)
    spec_and_sr = [0, 0, "24"] #make a list to be consistent with other methods


    where_to_find_good_audio = find_a_place_for_good_audio()

    save_spectro_image(where_to_find_good_audio)

    #return
    print ("############pred_for_api is working###################")

    debug = 1
    if (debug):
        with open("debug_dump2.txt", "w") as file1:
            # Writing data to a file
            file1.write("pred_for_api reporting \n")
            file1.write(f"spec_and_sr = {spec_and_sr}  \n")
            file1.write(f"where_to_find_good_audio = {where_to_find_good_audio}  \n")

    return spec_and_sr, where_to_find_good_audio



#############################################################
#            test some of the methods
#############################################################


def main( ):
    '''tester'''
    print (f"testing preprocess method")
    preprocess(num_spectrograms=2, num_speaker =0, debug=0)

    print (f"testing trainind method with model microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")

    train(enhancer="microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")



if __name__ == '__main__':
    main()
