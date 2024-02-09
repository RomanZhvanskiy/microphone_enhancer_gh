import os

# audio folders
audio_data_folder_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'audio_data')
audio_in_path = os.path.join(audio_data_folder_path, 'audio_in')
audio_out_path = os.path.join(audio_data_folder_path, 'audio_out')

##################  VARIABLES  ##################
GCP_PROJECT = "<your project id>" # TO COMPLETE
TRAINING_DATA_LOCATION = "use_local_data"
TRAINING_DATA_SUBFOLDER = "/Data/raw_data/VCTK-Corpus/wav48"
POSTPROCESSED_TRAINING_DATA_SUBFOLDER = "/Data/postprocessed_training_data"

MODEL_TYPE = "autoencoder_10_256"
MODEL_SUBFOLDER = "/pretrained_models/freshly_trained_model"
BAD_QUALITY_FILE  = "/Data/audio_data/audio_in/bad_quality_.wav"
GOOD_QUALITY_FILE = "/Data/audio_data/audio_out/good_quality.wav"
