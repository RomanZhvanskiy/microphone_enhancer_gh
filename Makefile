#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y microphone-enhancer || :
	@pip install -e .

run_preprocess:
	python -c 'from Back_end.interface.audioenhancer_local import preprocess; preprocess()'

run_train:
	python -c 'from Back_end.interface.audioenhancer_local import train; train()'

run_pred:
	python -c 'from Back_end.interface.audioenhancer_local import pred; pred()'

run_evaluate:
	python -c 'from Back_end.interface.audioenhancer_local import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate


## TESTS ##
api_test:
	uvicorn api.enhancer_api:app --reload &
	sleep 5
	streamlit run test_api.py
