## RUN API + FE LOCALLY ##
api_fe:
	uvicorn api.enhancer_api:app --reload &
	sleep 5
	streamlit run ../Front_end/test_api.py

## REMOVE UNNECESSARY FILES & FOLDERS ##
clean:
#	@rm -f */version.txt
#	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
#	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@rm -Rf audio_cache/*
#	@rm -Rf results

## INSTALL
install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U
