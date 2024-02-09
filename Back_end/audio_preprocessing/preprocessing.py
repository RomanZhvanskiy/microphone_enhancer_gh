"""This module does audio processing
    rev.0.1 --changed sampling rate to 48000 Hz,
        changed the way files are addressed
        removed the need for txt files
        reduced the resolution of MEL spectrogram from 512 to 256 frequency bands
"""

import librosa
import os
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from sklearn.utils import shuffle


def get_base_dir(debug = 0, working_in_google_colab = False):
    if (not working_in_google_colab):
        #should get something like:
        #/home/romanz/code/RomanZhvanskiy/microphone_enhancer_gh/raw_data/VCTK-Corpus
        #the preprocessing.py is located in
        #/home/romanz/code/RomanZhvanskiy/microphone_enhancer_gh/microphone_enhancer_gh/audio_preprocessing
        base_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = base_dir.replace("/microphone_enhancer_gh/audio_preprocessing", "/microphone_enhancer_gh/raw_data/VCTK-Corpus")
    else :
        base_dir = "/content/gdrive/MyDrive/Colab Notebooks/data_audio/VCTK-Corpus"
        if (debug): print (f"looking for training data here {base_dir}")

    return base_dir


###############################################################################
#                    plotting subroutines
###############################################################################

def plot_mel_spectrogram(spectrogram,sr,figsize=(8, 7), debug=0, ):
    """
    plot_mel_spectrogram makes a plot of a spectrogram

    Args:
        spectrogram: spectrogram to be plotted (in the form of 2D numpy array,
            where 1st dimension is n_mels, and 2nd dimensions is number of timesteps)
        sr: sampling rate
        figsize : size of the figure to plot
        debug=0: (optional) : if debug=1, messages will be printed during execution
            which can be helpful for debugging



    Returns:
        nothing

    Example use:
    plot_mel_spectrogram(x,sr, figsize=(8, 7))

    """
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    if (debug):
        print(f"power_to_db = {power_to_db}")
    plt.figure(figsize=figsize)
    librosa.display.specshow(power_to_db,
                            sr=sr,
                            x_axis='time',
                            y_axis='mel',
                            cmap='magma',
                            )
    plt.colorbar(label='dB')
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()

def plot_mel_spectrogram_of_waveform (y,sr):
    """
    plot_mel_spectrogram_of_waveform makes a plot of a spectrogram
    which it obtains by converting waveform to spectrogram

    Args:
        y: waveform to be plotted (in the form of 1D numpy array)
        sr: sampling rate


    Returns:
        nothing

    Example use:
    plot_mel_spectrogram(x,sr)

    """
    mel_signal = librosa.feature.melspectrogram(y=y,  sr=sr, n_mels=256, n_fft=2048)
    spectrogram = np.abs(mel_signal)
    plot_mel_spectrogram(spectrogram,sr, debug=0)
###############################################################################
#                    conversion between wav and sg subroutines
###############################################################################
def waveform_2_spectrogram (y,sr):
    """
    convert waveform to spectrogram

    Args:
        y: waveform to be converted (in the form of 1D numpy array)
        sr: sampling rate


    Returns:
        spectrogram

    Example use:
    spectrogram = waveform_2_spectrogram(x,sr)

    """
    mel_signal = librosa.feature.melspectrogram(y=y,  sr=sr, n_mels=256, n_fft=2048)
    #mel_signal = librosa.feature.melspectrogram(y=y,  sr=sr, n_mels=4, n_fft=32)
    spectrogram = np.abs(mel_signal)
    return spectrogram



def spectrogram_2_waveform (spectrogram,sr):
    """
    convert spectrogram to waveform

    Args:
        spectrogram: wavespectrogramform to be converted
        sr: sampling rate


    Returns:
        waveform

    Example use:
        waveform = spectrogram_2_waveform(spectrogram,sr)

    """

    waveform = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=2048)
    return waveform

###############################################################################
#                    writing to file subroutines
###############################################################################

def spectrogram_2_file (spectrogram,sr,filename="test_sound.wav"):
    import numpy as np
    import soundfile as sf

    """
    write spectrogram to file

    Args:
        spectrogram: spectrogram to be written to file
        sr: sampling rate
        filename: where to write the file


    Returns:
        nothing

    Example use:
    spectrogram = spectrogram_2_file (spectrogram,sr,filename)

    """
    #convert spectrogram to waveform
    waveform1 = spectrogram_2_waveform (spectrogram,sr)
    #write waveform to file
    sf.write(filename, waveform1, sr, subtype='PCM_24')

    return

def waveform_2_file (waveform1,sr,filename="test_sound.wav"):
    import numpy as np
    import soundfile as sf

    """
    write spectrogram to file

    Args:
        waveform1: waveform to be written to file
        sr: sampling rate
        filename: where to write the file


    Returns:
        nothing

    Example use:
    spectrogram = spectrogram_2_file (spectrogram,sr,filename)

    """
    #write waveform to file
    sf.write(filename, waveform1, sr, subtype='PCM_24')

    return

###############################################################################
#                    degrade quality subroutines
###############################################################################
def mel_spectrogram_remove_frequency(spectrogram, sr, remove_above=100000, remove_below=0, debug=0):
    """
    degrade_mel_spectrogram_remove_frequency removes frequencies above remove_above and below remove_below (in hz)
    which it obtains by converting waveform to spectrogram

    Args:
        spectrogram: spectrogram to be worked on (in the form of 2D numpy array)
        sr: sampling rate
        remove_above=100000: remove frequencies above this value
        remove_below=0: remove frequencies below this value
        debug=0: if debug=1, messages will be printed during execution
            which can be helpful for debugging

    Returns:
        degraded_spectrogram

    Example use:
        degraded_spectrogram = mel_spectrogram_remove_frequency(
            spectrogram,
            sr,
            remove_above=3000.0,
            remove_below=100.0,
            debug=0)

    """

    max_sampling_frequency = sr/2
    if (debug): print (f"max_sampling_frequency = {max_sampling_frequency }")

    n_mels = spectrogram.shape[0]
    if (debug): print (f"n_mels = {n_mels }")

    row_frequencies  = librosa.mel_frequencies(n_mels = n_mels, fmin=0.0, fmax = max_sampling_frequency)
    if (debug): print (f"row_frequencies = {row_frequencies }")
    row_frequencies_lst = row_frequencies.tolist()
    if (debug): print (f"row_frequencies_lst = {row_frequencies_lst }")

    #create array keep_frequencies where value 1 corresponds to keeping the frequency, and value 0 to dropping it
    keep_frequencies = np.empty_like(row_frequencies)

    for num_frequency, frequency in enumerate(row_frequencies_lst):
        #print (f"num_frequency = {num_frequency}, frequency = {frequency}")
        keep_frequencies[num_frequency] = 1
        if (frequency < remove_below):
            keep_frequencies[num_frequency] = 0
        if (frequency > remove_above):
            keep_frequencies[num_frequency] = 0

    #multiply column - wise original spectrogram with keep_frequencies vector to zero out some frequencies
    degraded_spectrogram  =  spectrogram*keep_frequencies[:, None]
    #print (f"keep_frequencies = {keep_frequencies}")

    return degraded_spectrogram

def sig(x):
 return 1/(1 + np.exp(-x))

def mel_spectrogram_add_noise(spectrogram, sr, relative_noise_level=0.1, add_above=0, add_below=100000, fade_width = 0.5,debug=0):
    """
    degrade_mel_spectrogram_remove_frequency removes frequencies above remove_above and below remove_below (in hz)
    which it obtains by converting waveform to spectrogram

    Args:
        spectrogram: spectrogram to be worked on (in the form of 2D numpy array)
        sr: sampling rate
        relative_noise_level: the level of added noise as a fraction of mean loudness in the spectrogram
        add_above=0: add noize above this value
        add_below=100000:      add noize below this value
        fade_width=0.5: how wide should be noize fade
        debug=0: if debug=1, messages will be printed during execution
            which can be helpful for debugging

    Returns:
        degraded_spectrogram

    Example use:
        degraded_spectrogram = mel_spectrogram_add_noise(
            spectrogram,
            sr,
            add_above=100.0,
            add_below=3000.0,
            relative_noise_level=0.00001,
            debug=0)

    """
    mean_loudness = spectrogram.mean()
    noise_level = mean_loudness*relative_noise_level
    max_sampling_frequency = sr/2
    n_mels = spectrogram.shape[0]
    row_frequencies  = librosa.mel_frequencies(n_mels = n_mels, fmin=0.0, fmax = max_sampling_frequency)
    row_frequencies_lst = row_frequencies.tolist()

    #create array noise_frequencies where value 1 corresponds to keeping the frequency, and value 0 to dropping it

    noise_spectrogram =   np.zeros_like(spectrogram) # empty spectrogram with frequency


    for num_frequency, frequency in enumerate(row_frequencies_lst):
        #print (f"num_frequency = {num_frequency}, frequency = {frequency}")
        if (frequency >= add_above and frequency <= add_below):
            #make  a graduall "activation" with sigmoid
            how_far_above_noize_floor = frequency - add_above
            how_far_below_noize_ceiling = add_below - frequency
            activation_above = (sig(how_far_above_noize_floor/(add_above*fade_width))-0.5)
            activation_below = (sig(how_far_below_noize_ceiling/(add_below*fade_width)) - 0.5)
            noise_spectrogram[num_frequency, :] = noise_level*activation_above*activation_below




    degraded_spectrogram  =  spectrogram + noise_spectrogram


    if (debug):
        print (f"mean_loudness         = {mean_loudness        }")
        print (f"noise_level           = {noise_level         }")
        print (f"degraded_spectrogram  = {degraded_spectrogram}")

    return degraded_spectrogram


def mel_spectrogram_remove_quiet_sounds (spectrogram, sr,  remove_below=0.01, debug=0):
    """
    degrade_mel_spectrogram_remove_frequency removes frequencies above remove_above and below remove_below (in hz)
    which it obtains by converting waveform to spectrogram

    Args:
        spectrogram: spectrogram to be worked on (in the form of 2D numpy array)
        sr: sampling rate
        remove_below: relative (to average) loudness of sounds which will be removed
        debug=0: if debug=1, messages will be printed during execution
            which can be helpful for debugging

    Returns:
        degraded_spectrogram

    Example use:
        degraded_spectrogram = mel_spectrogram_add_noise(
            spectrogram,
            sr,
            remove_below=0.01,
            debug=0)

    """


    mean_loudness = spectrogram.mean()
    cut_off_below_this_intensity = mean_loudness*remove_below
    if (debug):
        print (f"spectrogram          = {spectrogram        }")
        print (f"sr          = {sr        }")
        print (f"remove_below          = {remove_below        }")
        print (f"cut_off_below_this_intensity          = {cut_off_below_this_intensity        }")

    #create array noise_frequencies where value 1 corresponds to keeping the frequency, and value 0 to dropping it

    degraded_spectrogram = spectrogram
    degraded_spectrogram[degraded_spectrogram < cut_off_below_this_intensity] = 0

    if (debug):
        print (f"mean_loudness          = {mean_loudness        }")
        print (f"degraded_spectrogram  = {degraded_spectrogram}")

    return degraded_spectrogram

def degrade_quaity(spectrogram, sr, upper_limit=3000.0, lower_limit=100.0, insensitive_level = 0.5,relative_noise_level=0.1, debug=0):
    degraded_spectrogram = mel_spectrogram_remove_frequency(
            spectrogram,
            sr,
            remove_above=upper_limit,
            remove_below=lower_limit,
            debug=debug)


    #remove quiet sounds  (our simulated bad microphone cannot capture quiet sounds)
    degraded_spectrogram = mel_spectrogram_remove_quiet_sounds (
            degraded_spectrogram,
            sr,
            remove_below=insensitive_level,
            debug=debug)

    #add noise (our simulated bad microphone also captures noize)
    degraded_spectrogram = mel_spectrogram_add_noise(degraded_spectrogram,
            sr,
            relative_noise_level=relative_noise_level,
            add_above=lower_limit,
            add_below=upper_limit,
            debug=debug)
    return degraded_spectrogram

###############################################################################
#                    subroutines to load data from HDD
###############################################################################

def load_wav (where_to_find_audio):
    """
    load_wav reads a .wav file into  a waveform `x` and also reads a sampling rate

    Args:
        where_to_find_audio: string


    Returns:
        X is a 1-D numpy array containing a waveform
        sr is a floating point number which provides sampling rate.

    Example use:
    x, sr  = get_speech()

    """

    speech_path = where_to_find_audio
    SAMPLING_RATE = 48000 #22050 - old value
    x, sr = librosa.load(speech_path, sr=SAMPLING_RATE) # Load the audio as a waveform `x`
    return x, sr




def get_speech(speaker_id = -1 , passage_id = None, where_to_get_training_data="not specified", working_in_google_colab = False):
    """
    get_speech reads a .wav file into  a waveform `x` and also reads a sampling rate

    Args:
        speaker_id: integer. If not specified, a random speaker_id will be selected
        passage_id: integer. If not specified, a random passage_id will be selected


    Returns:
        X is a 1-D numpy array containing a waveform
        sr is a floating point number which provides sampling rate.

    Example use:
    x, sr  = get_speech()

    """

    if (where_to_get_training_data == "not specified"):
        BASE_DIR = get_base_dir(working_in_google_colab=working_in_google_colab)
        AUDIO_DIR = os.path.join(BASE_DIR, 'wav48')
    else:
        AUDIO_DIR = where_to_get_training_data

    SAMPLING_RATE = 48000 #22050 - old value
    MAX_DURATION = 8
    SR_DOWNSAMPLE = 2
    LOAD_CHECKPOINT = False

    speaker_ids = sorted(os.listdir(AUDIO_DIR))

    if (speaker_id == -1):
        speaker_id = np.random.choice(speaker_ids)

    print('number of speakers is', len(speaker_ids))

    if not passage_id:

        speaker_id = np.random.choice(os.listdir(AUDIO_DIR))
        print (f"speaker_id ={speaker_id}" )
        speaker_passage_path = AUDIO_DIR + "/"+ speaker_id
        passage_id = np.random.choice(os.listdir(speaker_passage_path))
        print (f"passage_id ={passage_id}" )
        speech_path = speaker_passage_path  + "/"+ passage_id

    speech_path = os.path.join(AUDIO_DIR, speaker_id, passage_id + '.wav')
    speech_path = speech_path.replace(".wav.wav",".wav")
    x, sr = librosa.load(speech_path, sr=SAMPLING_RATE) # Load the audio as a waveform `x`
    return x, sr




def get_all_speech_as_one_mel(num_spectrograms=10000, num_speaker = 0, where_to_get_training_data="not specified", random_state=1, debug = 0,  working_in_google_colab = False):
    """
    get_all_speech_as_one_mel reads all .wav files into  a waveform;
    converts each waveform into MEL spectrogram;
    adds up all MEL spectrograms and returns one big MEL spectrogram


    Args:
        num_spectrograms: integer. If not specified, all spectrograms will be returned for the speaker
        num_speaker: integer.
        random_state: integer. this will fix the random spectrogram selection
        debug=0: (optional) : if debug=1, messages will be printed during execution
            which can be helpful for debugging

    Returns:
        X is a nD numpy array containing all spectrograms appended together with the following dimensions (example):
        ( 256, 100000)
        where
            256 - number of spectral bands in each spectrogram
            100000  - number of timesteps in each spectrogram

        sr is a floating point number which provides sampling rate.

    Example use:
    x, sr  = get_speech()

    """
    if (debug) : print (f"where_to_get_training_data={where_to_get_training_data}")

    if (where_to_get_training_data == "not specified"):
        BASE_DIR = get_base_dir(working_in_google_colab=working_in_google_colab)
        AUDIO_DIR = os.path.join(BASE_DIR, 'wav48')
    else:
        AUDIO_DIR = where_to_get_training_data

    #BASE_DIR = get_base_dir(working_in_google_colab=working_in_google_colab)
    #TXT_DIR = os.path.join(BASE_DIR, 'txt')
    #AUDIO_DIR = os.path.join(BASE_DIR, 'wav48')
    SAMPLING_RATE = 48000
    MAX_DURATION = 8
    SR_DOWNSAMPLE = 2
    LOAD_CHECKPOINT = False

    #check how many speakers are there in the folder

    speaker_ids = sorted(os.listdir(AUDIO_DIR))
    #if(debug, working_in_google_colab = False):
    #    print (f"speaker_ids = {speaker_ids}")


    #assign a sequential number to each speaker


    dict_speakers = {}
    for speaker_n, speaker_id in enumerate(speaker_ids):
        #dict_speakers.append({speaker_n:speaker_id})
        dict_speakers.update({speaker_n:speaker_id})


    #look for the speaker with correct sequential number

    if not (num_speaker in dict_speakers):
        print(f"num_speaker = {num_speaker} is not in data. Please select a different num_speaker")
        return
    else:
        print(f"num_speaker = {num_speaker}; folder = {dict_speakers[num_speaker]}")

    #check how many recorded waveforms are in the folder for the selected speaker

    speaker_passage_path = AUDIO_DIR + "/"+ dict_speakers[num_speaker]

    path_ids = sorted(os.listdir(speaker_passage_path))
    #if(debug):
    #    print (f"path_ids = {path_ids}")

    if (num_spectrograms > len(path_ids)):
        print (f"Number of recorded waveforms for speaker #{num_speaker} is less than requested.")
        print (f"Reading all available waveforms, {len(path_ids)} in total.")
        num_spectrograms = len(path_ids)

    #randomly select from all available waveforms the num_spectrograms

    all_available_waveforms = range(0, num_spectrograms)
    all_available_waveforms = shuffle(all_available_waveforms, random_state=random_state)
    selected_waveforms = all_available_waveforms[0:num_spectrograms]

    #read all selected_waveforms

    for waveform_no, i in enumerate(selected_waveforms):
        selected_waveform_path = speaker_passage_path + "/" + path_ids[i]
        wf, sr = librosa.load(selected_waveform_path)


        #convert each waveform to spectrogram
        sg = waveform_2_spectrogram (wf, sr)
        print (f"sg.shape = {sg.shape} ")
        list_sg = np.transpose(sg).tolist()  #transpose and convert np array to list for easy extending;
        if (waveform_no == 0):
            list_total_sg =  list_sg
        else:
            list_total_sg.extend(list_sg)

    #print (f"list_total_sg = {list_total_sg} ")

    #convert list back to np array and transpose to retain original dimensions

    total_sg  =  np.transpose(np.asarray(list_total_sg))


    print (f"total_sg.shape = {total_sg.shape} ")
    return total_sg, sr


###############################################################################
#                    train test split
###############################################################################

def split_spectrogram_in_train_and_test (spectrogram,
                                         test_ratio = 0.2,
                                         debug = 0, working_in_google_colab = False):

    """
     split  spectrogram in train and test

    Args:
        spectrogram: spectrogram to be split in train and test
        test_ratio: what fraction of the spectrogram will be test (the rest will be train)

    Returns:
        train_sg, test_sg

    Example use:
    train_sg, test_sg = split_spectrogram_in_train_and_test (spectrogram,0.2)

    """
    n_timesteps = spectrogram.shape[1]
    n_timesteps_test = int(test_ratio*n_timesteps)
    n_timesteps_train = n_timesteps - n_timesteps_test

    if (debug):
        print (f"n_timesteps       = {n_timesteps      }")
        print (f"n_timesteps_test  = {n_timesteps_test }")
        print (f"n_timesteps_train = {n_timesteps_train}")
    train_sg = spectrogram.copy()
    train_sg = train_sg[:, 0:n_timesteps_train]
    test_sg = spectrogram.copy()
    test_sg = test_sg[:, n_timesteps_train:n_timesteps]
    return train_sg, test_sg


#test some of the methods
def main( working_in_google_colab = False):
    '''tester'''
    print (f"get_base_dir(working_in_google_colab) = {get_base_dir(working_in_google_colab=working_in_google_colab)}")

if __name__ == '__main__':
    main()
