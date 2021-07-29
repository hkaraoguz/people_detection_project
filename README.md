# people_detection_project

This is an end-to-end pipeline for detecting, tracking, storing and viewing people from surveillance cameras. The whole pipeline can be deployed using Docker. GPU mode is supported when available.

## Requirements
* docker-compose >= 1.28

## Running the pipeline
* First, it is recommended to setup `.env` file for object detection module. Open the `detection.env` file under `object_detection` folder in order to set the `camera_url` parameter.
* Based on the setup you have, you can run any of the docker-compose files
### Full-pipeline w/o GPU
Run 
```
docker-compose up --force-recreate
```
This will launch four containers and create a custom network. `--force-recreate` flag ensures that there is no residuals left from previous builds.
* `mongo` container will run an instance of mongodb 
* `object_detection` container will run the object detector in `torchvision` along with person tracking algorithm using [sort](https://github.com/abewley/sort). The tracked people are stored in mongodb with timestamp information as well as a screenshot from camera image.
* `api` container will run an instance of REST API interface for fetching/filtering the detection results from mongodb.
* `html` container will run an instance of `nginx` that serves `index.html` package for visualizing the results. <br>

You can navigate to `localhost:8080` in your browser to visualize the results
### Full pipeline with GPU
```
docker-compose up -f docker-compose_gpu.yaml --force-recreate
```
### Only person detection and database with GPU
```
docker-compose up -f docker-compose_detection_gpu --force-recreate
```

### Only person detection and database w/o GPU
```
docker-compose up -f docker-compose_detection --force-recreate
```
### Pipeline for only viewing the results
```
docker-compose up -f docker-compose_view --force-recreate
```