"""
This module implements api for getting model predictions with the help of FastAPI:
https://fastapi.tiangolo.com
"""
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from api.api_func import *
from params import *
from hugging_models.hugrestore import dns4_16k, wham_16k
from interface.audioenhancer_local import  pred_for_api
app = FastAPI()

app.mount("/audio_in", StaticFiles(directory=audio_in_path), name="audio_in")
app.mount("/audio_out", StaticFiles(directory=audio_out_path), name="audio_out")

@app.get('/') # root endpoint
def index():
    return {'It works!': True}

@app.get("/audio_in_list") # list files in audio_in deirectory
def audio_in():
    return list_path(audio_in_path)

@app.get("/audio_out_list") # list files in audio_out deirectory
def audio_in():
    return list_path(audio_out_path)

@app.post("/upload_file")
async def upload_file(enhancer: str = Form(...), file: UploadFile = File(...)):
    # temp file logic
    temp = NamedTemporaryFile(delete=False)
    with temp as f:
        f.write(file.file.read())

    if enhancer == "speechbrain/sepformer-wham16k-enhancement":
        spec_aud_sr, cleaned_path = wham_16k(temp.name, from_fs=False)
    elif enhancer == "speechbrain/sepformer-dns4-16k-enhancement":
        spec_aud_sr, cleaned_path = dns4_16k(temp.name, from_fs=False)
    elif enhancer == "microphone_enhancer_gh/autoencoder_10_256":
        spec_aud_sr, cleaned_path = pred_for_api(where_to_find_bad_audio=temp.name, enhancer="microphone_enhancer_gh/autoencoder_10_256")


    elif enhancer == "microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1":
        #python -c 'from Back_end.interface.audioenhancer_local import pred; pred( enhancer="microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")'
        #import ipdb; ipdb.set_trace()
        spec_aud_sr, cleaned_path = pred_for_api(where_to_find_bad_audio=temp.name, enhancer="microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1")
        #import ipdb; ipdb.set_trace()

    return {"cleaned_file_name": os.path.basename(cleaned_path),
            "sample rate": spec_aud_sr[2],
            #"audio": cleaned_audio,
            }
