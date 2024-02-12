## BUILD DOCKER IMAGE ##
docker_build_dev:
	docker build --tag=micenhancer_api:dev .

docker_run:
	docker run -it -e PORT=8000 -p 8000:8000 micenhancer_api:dev

## RUN API + FE LOCALLY ##
api_fe:
	uvicorn api.enhancer_api:app --reload &
	sleep 5
	streamlit run ../Front_end/test_api.py

api_fe2:
	uvicorn api.enhancer_api_connected_to_backend2:app --reload &
	sleep 5
	streamlit run ../Front_end/test_api.py

api:
	uvicorn api.enhancer_api:app --reload &

api2:
	uvicorn api.enhancer_api_connected_to_backend2:app --reload &

fe:
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

install_editable:
	@pip install -e .

install:
	@pip install . -U

uninstall:
	@pip uninstall -y microphone-enhancer
