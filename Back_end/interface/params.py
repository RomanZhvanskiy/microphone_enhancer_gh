import os
import numpy as np

##################  VARIABLES  ##################
GCP_PROJECT = "<your project id>" # TO COMPLETE
TRAINING_DATA_LOCATION = "use_local_data"
TRAINING_DATA_SUBFOLDER = "training_data"
MODEL_TYPE = "autoencoder_10_256"
MODEL_SUBFOLDER = "freshly_trained_model"
BAD_QUALITY_FILE  = "IO_files/bad_quality_.wav"
GOOD_QUALITY_FILE = "IO_files/good_quality.wav"




#GCP_PROJECT_WAGON = "wagon-public-datasets"  # TO COMPLETE
#BQ_DATASET = "taxifare"
#BQ_REGION = "EU"
#MODEL_TARGET = "local"
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
#
#COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
#
#DTYPES_RAW = {
#    "fare_amount": "float32",
#    "pickup_datetime": "datetime64[ns, UTC]",
#    "pickup_longitude": "float32",
#    "pickup_latitude": "float32",
#    "dropoff_longitude": "float32",
#    "dropoff_latitude": "float32",
#    "passenger_count": "int16"
#}
#
#DTYPES_PROCESSED = np.float32
