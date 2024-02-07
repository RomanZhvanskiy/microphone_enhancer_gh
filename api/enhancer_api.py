"""
This module implements api for getting model predictions with the help of FastAPI:
https://fastapi.tiangolo.com
"""

from fastapi import FastAPI, UploadFile, File, Form
#from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from api.api_func import *
from params import *
from hugging_models.hugrestore import dns4_16k, wham_16k

app = FastAPI()

#app.mount("/audio_in", StaticFiles(directory=audio_in_path), name="audio_in")
#app.mount("/audio_out", StaticFiles(directory=audio_out_path), name="audio_out")

@app.get('/') # root endpoint
def index():
    return {'It works!': True}

@app.get('/enhance') # enhancer endpoint
def index():
    return {'Enhancer endpont': True}

@app.get("/audio_in") # list files in audio_in deirectory
def audio_in():
    return list_path(audio_in_path)

@app.get("/audio_out") # list files in audio_out deirectory
def audio_in():
    return list_path(audio_out_path)

@app.post("/upload_file")
async def upload_file(enhancer: str = Form(...), file: UploadFile = File(...)):
    #print(type(file.file))
    #ala, audio = dns4_16k(file)
    #print(file.file.read())
    #import soundfile as sf
    #data, sr = sf.read(file.file)
    print(enhancer)
    temp = NamedTemporaryFile(delete=False)
    #print(file.file.read())
    with temp as f:
        f.write(file.file.read())
    #print(temp)
    if enhancer == "speechbrain/sepformer-wham16k-enhancement":
        spec_aud_sr, cleaned_path = dns4_16k(temp.name, from_fs=False)
    elif enhancer == "speechbrain/sepformer-dns4-16k-enhancement":
        spec_aud_sr, cleaned_path = wham_16k(temp.name, from_fs=False)
    elif enhancer == "speechbrain/sepformer-dns4-16k-enhancement":
        False #spec_aud_sr, cleaned_path = dns4_16k(temp.name, from_fs=False)

    #print(collected[2])
    #sf.write(data, sr)
    return {"cleaned_file_name": cleaned_path,
            "sample rate": spec_aud_sr[2],
            #"audio": cleaned_audio,
            }
