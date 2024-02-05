from hugging_models import hugrestore

# test for 'hugging_models' module
#
# please put noisy file into 'audio_data/audio_in' and
# update 'input_audio_name' variable accordingly

input_audio_name = 'book_00000_chp_0009_reader_06709_17_seg_live_phone_1'

hugrestore.wham_16k(input_audio_name)
hugrestore.dns4_16k(input_audio_name)
