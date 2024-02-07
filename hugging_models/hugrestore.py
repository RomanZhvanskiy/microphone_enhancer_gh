"""
This module cleans the audio file using SpeechBrain pre-trained models from Hugging Face:
    https://huggingface.co/speechbrain
    wham_16k: https://huggingface.co/speechbrain/sepformer-wham16k-enhancement
    dns4_16k: https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement
"""

#from scipy.io.wavfile import write
#import numpy as np
import os
import sys

import torchaudio
import librosa
#import soundfile as sf
from speechbrain.pretrained import SepformerSeparation as separator

from audio_preprocessing import preprocessing as pp


def check_folders(input_file:str):
    """
    Checks if audio directories exist and if not - creates them.
    Checks if nooisy file exists and if not - exits.
    """
    audio_data_folder = 'audio_data'
    audio_data_folder_path = os.path.join(os.path.dirname(__file__), '..', audio_data_folder)
    audio_in_path = os.path.join(audio_data_folder_path, 'audio_in')
    audio_out_path = os.path.join(audio_data_folder_path, 'audio_out')

    if not os.path.exists(audio_data_folder_path):
        os.makedirs(audio_data_folder_path)

    if not os.path.exists(audio_in_path):
        os.makedirs(audio_in_path)

    if not os.path.exists(audio_out_path):
        os.makedirs(audio_out_path)

    if not os.path.isfile(os.path.join(audio_in_path, input_file + '.wav')):

        # need to change handling of this exception to something reasonabel (now stops the program)
        sys.exit(f'File {input_file}.wav does not exist, please make sure the file is in the '\
               '"audio_in" folder and if the name (without .wav extension) is spelled correctly')

    return audio_data_folder, audio_in_path, audio_out_path


def wham_16k(input_file:str):
    """
    Cleans the audio with 'speechbrain/sepformer-wham16k-enhancement' and saves it to
    'audio_data/audio_out' folder. Returns both spectogram and audio array.

        Parameters:
            input_file (str): Input wav audio file name without extension.
            The file must be placed into 'audio_data/audio_in' folder.
    """
    audio_data_folder, audio_in_path, audio_out_path = check_folders(input_file)

    full_input_file = os.path.join(audio_in_path, input_file + '.wav')
    full_output_file = os.path.join(audio_out_path, input_file + '-wham16k-res' + '.wav')
    #print(full_input_file, full_output_file)
    model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement",
                                   savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(full_input_file) # autoconverts to 16kHz
    #print(est_sources[:,:,0].numpy())
    #write(full_output_file, sample_rate, est_sources[:,:,0].numpy())
    torchaudio.save(full_output_file, est_sources[:, :, 0].detach().cpu(), 16000)
    print(f"File restored with 'speechbrain/sepformer-wham16k-enhancement' and saved as "\
          f"'{input_file}-wham16k-res.wav' in '{audio_data_folder}/audio_out' folder.")

    return wav_to_spectro(full_output_file, 16000)


def dns4_16k(input_file:str):
    """
    Cleans the audio with 'speechbrain/sepformer-dns4-16k-enhancement' and saves it to
    'audio_data/audio_out' folder. Returns both spectogram and audio array.

        Parameters:
            input_file (str): Input wav audio file name without extension.
            The file must be placed into 'audio_data/audio_in' folder.
    """
    audio_data_folder, audio_in_path, audio_out_path = check_folders(input_file)

    full_input_file = os.path.join(audio_in_path, input_file + '.wav')
    full_output_file = os.path.join(audio_out_path, input_file + '-dns4-16k-res' + '.wav')
    model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                   savedir='pretrained_models/sepformer-dns4-16k-enhancement')
    est_sources = model.separate_file(full_input_file) # autoconverts to 16kHz
    torchaudio.save(full_output_file, est_sources[:, :, 0].detach().cpu(), 16000)
    #sf.write(full_output_file, est_sources[:, :, 0][0], 16000, subtype='PCM_16')
    print(f"File restored with 'speechbrain/sepformer-dns4-16k-enhancement' and saved as "\
          f"'{input_file}-dns4-16k-res.wav' in '{audio_data_folder}/audio_out' folder.")

    return wav_to_spectro(full_output_file, 16000)


def wav_to_spectro(wav_file_path: str, sr: int):
    """
    Converts wav file to a spectrogram array and returns both spectogram and audio array.

        Parameters:
            wav_file_path (str): Full wav audio file path.
            sr (int): Sample rate of wav file.
    """
    x, sr = librosa.load(wav_file_path, sr=sr)
    spectrogram = pp.waveform_2_spectrogram (x,sr)
    return spectrogram, x
