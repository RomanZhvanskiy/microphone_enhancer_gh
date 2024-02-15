## BUILD LOCAL DOCKER IMAGE ##
docker_build_dev:
	docker build --tag=micenhancer_api:dev .

docker_run:
	docker run -it -e PORT=8000 -p 8000:8000 micenhancer_api:dev


## DOCKER FOR THE CLOUD !! IMPORTANT - CHANGE lofty-door-406712 TO YOU GOOGLE PROJECT NAME

docker_cloud_build:
	docker build -t europe-west6-docker.pkg.dev/lofty-door-406712/micenhancer/micenhancer_api:test .

#run locally to test
docker_cloud_run_loc:
	docker run -e PORT=8000 -p 8000:8000 europe-west6-docker.pkg.dev/lofty-door-406712/micenhancer/micenhancer_api:test

docker_cloud_push:
	docker push europe-west6-docker.pkg.dev/lofty-door-406712/micenhancer/micenhancer_api:test

docker_cloud_run:
	gcloud run deploy --image europe-west6-docker.pkg.dev/lofty-door-406712/micenhancer/micenhancer_api:test --memory 4Gi --region europe-west6 --min-instances 1

docker_cloud_stop:
	gcloud run services delete micenhancerapi


## RUN API + FE LOCALLY ##
api_fe:
	uvicorn api.enhancer_api:app --reload &
	sleep 5
	streamlit run ../Front_end/test_api.py --server.maxUploadSize 5

api_fe2:
	uvicorn api.enhancer_api_connected_to_backend2:app --reload &
	sleep 5
	streamlit run ../Front_end/test_api.py --server.maxUploadSize 5

api:
	uvicorn api.enhancer_api:app --reload &

api2:
	uvicorn api.enhancer_api_connected_to_backend2:app --reload &

fe:
	streamlit run ../Front_end/test_api.py --server.maxUploadSize 5

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
