"""
This module implements api for getting model predictions with the help of FastAPI:
https://fastapi.tiangolo.com
"""
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from api.api_func import *
from params import *
from hugging_models.hugrestore import dns4_16k, wham_16k
from speechbrain.pretrained import SepformerSeparation as separator

app = FastAPI()
app.state.loaded_dns4 = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                               savedir='../Data/pretrained_models/sepformer-dns4-16k-enhancement')
app.state.loaded_wham = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement",
                                               savedir='../Data/pretrained_models/sepformer-wham16k-enhancement')

# mountpoints to serve files
app.mount("/audio_in", StaticFiles(directory=audio_in_path), name="audio_in")
app.mount("/audio_out", StaticFiles(directory=audio_out_path), name="audio_out")

@app.get('/') # root endpoint
def index():
    return {'It works!': True}

@app.get('/enhance') # enhancer endpoint
def index():
    return {'Enhancer endpont': True}

@app.get("/audio_in_list") # list files in audio_in deirectory
def audio_in():
    return list_path(audio_in_path)

@app.get("/audio_out_list") # list files in audio_out deirectory
def audio_in():
    return list_path(audio_out_path)

@app.post("/upload_file") # upload file to enhance
def upload_file(enhancer: str = Form(...), file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    with temp as f:
        f.write(file.file.read())
    if enhancer == "speechbrain/sepformer-wham16k-enhancement":
        spec_aud_sr, cleaned_path = wham_16k(app.state.loaded_wham, temp.name, from_fs=False)
    elif enhancer == "speechbrain/sepformer-dns4-16k-enhancement":
        spec_aud_sr, cleaned_path = dns4_16k(app.state.loaded_dns4, temp.name, from_fs=False)
    elif enhancer == "NOT IMPLEMENTED YET: microphone_enhancer_gh":
        False #spec_aud_sr, cleaned_path = dns4_16k(temp.name, from_fs=False)

    return {"cleaned_file_name": os.path.basename(cleaned_path),
            "sample rate": spec_aud_sr[2],
           }
