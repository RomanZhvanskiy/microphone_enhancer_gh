"""
This module cleans the audio file using SpeechBrain pre-trained models from Hugging Face:
    https://huggingface.co/speechbrain
    wham_16k: https://huggingface.co/speechbrain/sepformer-wham16k-enhancement
    dns4_16k: https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement
"""
import os
import sys
import shutil

import torchaudio
import librosa
from speechbrain.pretrained import SepformerSeparation as separator

from audio_preprocessing import preprocessing as pp
from params import *


def check_folders():
    """
    Checks if audio directories exist and if not - creates them.
    """
    if not os.path.exists(audio_data_folder_path):
        os.makedirs(audio_data_folder_path)
    if not os.path.exists(audio_in_path):
        os.makedirs(audio_in_path)
    if not os.path.exists(audio_out_path):
        os.makedirs(audio_out_path)

    return None


def wham_16k(preloaded_model, input_file:str, from_fs:bool=True):
    """
    Cleans the audio with 'speechbrain/sepformer-wham16k-enhancement'. Returns:
    spectorgramm, audio array, sampling rate and path to resulting file.

    Parameters:
        from_fs (bool): True if original file placed to audio_data/audio_in
                        False if path to temp file is passed instead
        input_file (str): Input wav audio file name or full path to temp file
    """
    if from_fs:
        check_folders()
        if os.path.isfile(os.path.join(audio_in_path, input_file + '.wav')):
            full_input_file = os.path.join(audio_in_path, input_file + '.wav')
            full_output_file = os.path.join(audio_out_path, input_file + '-wham_16k-res' + '.wav')
        else:
            sys.exit(f'File {input_file}.wav does not exist')
    else:
        full_input_file = input_file
        full_output_file = os.path.join(audio_out_path, os.path.basename(input_file) + '-wham_16k-res' + '.wav')

    shutil.copyfile(os.path.join(full_input_file), os.path.join(audio_in_path, os.path.basename(input_file) + '-wham_16k-orig' + '.wav'))
    est_sources = preloaded_model.separate_file(full_input_file) # autoconverts to 16kHz
    torchaudio.save(full_output_file, est_sources[:, :, 0].detach().cpu(), 16000)

    return wav_to_spectro(full_output_file, 16000), full_output_file


def dns4_16k(preloaded_model, input_file:str, from_fs:bool=True):
    """
    Cleans the audio with 'speechbrain/sepformer-dns4-16k-enhancement'. Returns:
    spectorgramm, audio array, sampling rate and path to resulting file.

    Parameters:
        from_fs (bool): True if original file placed to audio_data/audio_in
                        False if path to temp file is passed instead
        input_file (str): Input wav audio file name or full path to temp file
    """
    if from_fs:
        check_folders()
        if os.path.isfile(os.path.join(audio_in_path, input_file + '.wav')):
            full_input_file = os.path.join(audio_in_path, input_file + '.wav')
            full_output_file = os.path.join(audio_out_path, input_file + '-dns4-16k-res' + '.wav')
        else:
            sys.exit(f'File {input_file}.wav does not exist')
    else:
        full_input_file = input_file
        full_output_file = os.path.join(audio_out_path, os.path.basename(input_file) + '-dns4-16k-res' + '.wav')

    shutil.copyfile(os.path.join(full_input_file), os.path.join(audio_in_path, os.path.basename(input_file) + '-dns4_16k-orig' + '.wav'))
    est_sources = preloaded_model.separate_file(full_input_file) # autoconverts to 16kHz
    torchaudio.save(full_output_file, est_sources[:, :, 0].detach().cpu(), 16000)

    return wav_to_spectro(full_output_file, 16000), full_output_file


def wav_to_spectro(wav_file_path: str, sr: int):
    """
    Converts wav file to a spectrogram array and returns both spectogram and audio array.

        Parameters:
            wav_file_path (str): Full wav audio file path.
            sr (int): Sample rate of wav file.
    """
    audio_arr, sr = librosa.load(wav_file_path, sr=sr)
    spectrogram = pp.waveform_2_spectrogram (audio_arr, sr)

    return spectrogram, audio_arr, sr
