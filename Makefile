#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y microphone-enhancer || :
	@pip install -e .

run_preprocess:
	python -c 'from Back_end.interface.audioenhancer_local import preprocess; preprocess()'

run_train:
	python -c 'from Back_end.interface.audioenhancer_local import train; train(epochs=100 )'

run_train_2:
	python -c 'from Back_end.interface.audioenhancer_local import train; train(epochs=2, enhancer="microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")'


run_pred:
	python -c 'from Back_end.interface.audioenhancer_local import pred; pred()'
run_pred_2:
	python -c 'from Back_end.interface.audioenhancer_local import pred; pred( enhancer="microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")'
#run_evaluate:
#	python -c 'from Back_end.interface.audioenhancer_local import evaluate; evaluate()'#

#run_all: run_preprocess run_train run_pred run_evaluate

## REMOVE UNNECESSARY FILES & FOLDERS ##
clean:
#	@rm -f */version.txt
#	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
#	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@rm -Rf audio_cache
#	@rm -Rf results
