import os
import urllib.request
import requests
from PIL import Image
import streamlit as st

url = "http://127.0.0.1:8000/" # "https://micenhancerapi-3t3dih6maa-oa.a.run.app/", "http://127.0.0.1:8000/"
upload_url = url + 'upload_file'
serve_out_url = url + 'audio_out'
serve_in_url = url + 'audio_in'

st.subheader('Microphone quality enhancer')
st.caption('enhance the quality of your microphone')
st.divider()

enhancer = st.radio("Select enhancer:",
                    ["speechbrain/sepformer-dns4-16k-enhancement",
                     "speechbrain/sepformer-wham16k-enhancement",
                     "microphone_enhancer_gh/autoencoder_10_256",
                     "microphone_enhancer_gh/conv_autoencoder_16_32_64_32_16_1"],
                    index=0,
                    )
uploaded_file = st.file_uploader("Choose a noisy audio file (.wav):", type='wav')

st.divider()

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.write("Uploaded noisy audio:")
    with col2:
        st.write(f"Audio cleaned with {enhancer}:")
    col3, col4 = st.columns(2, gap='medium')
    with col3:
        st.audio(uploaded_file)
    file = {'file': uploaded_file}
    params = {'enhancer': enhancer}
    cleaned = requests.post(upload_url,
                            files=file,
                            data=params).json()
    with col4:
        req = urllib.request.urlopen(os.path.join(serve_out_url, cleaned['cleaned_file_name'])).read()
        st.audio(req)
        #st.audio(os.path.join(serve_out_url, cleaned['cleaned_file_name']), format="audio/wav")
    col5, col6 = st.columns(2, gap='medium')
    with col5:
        with Image.open(requests.get((os.path.join(serve_in_url, cleaned['cleaned_file_name']+'.jpg')), stream=True).raw) as image:
            st.image(image)
        #st.image(os.path.join(serve_in_url, cleaned['cleaned_file_name'])+'.jpg')
    with col6:
        with Image.open(requests.get((os.path.join(serve_out_url, cleaned['cleaned_file_name']+'.jpg')), stream=True).raw) as image:
            st.image(image)
        #st.image(os.path.join(serve_out_url, cleaned['cleaned_file_name'])+'.jpg')
