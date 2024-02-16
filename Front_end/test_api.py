import os
import time
import requests
from PIL import Image
import streamlit as st

url = "https://micenhancerapi-3t3dih6maa-oa.a.run.app/" # "https://micenhancerapi-3t3dih6maa-oa.a.run.app/", "http://127.0.0.1:8000/"
upload_url = url + 'upload_file'
serve_out_url = url + 'audio_out'
serve_in_url = url + 'audio_in'

col5, col6 = st.columns([10,1], gap='medium')
with col5:
    st.subheader('Microphone quality enhancer')
    st.caption('enhance the quality of your microphone')
with col6:
    with Image.open(requests.get('https://raw.githubusercontent.com/gvaa/microphone_enhancer_gh_fe/master/Front_end/microphone-162205_1280.png', stream=True).raw) as micro:
        st.image(micro, width = 40)

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
    file = {'file': uploaded_file}
    params = {'enhancer': enhancer}
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.markdown(f'<p style="background-color:#fcefec;color:#972007;border-radius:7px;"><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Uploaded noisy audio:<br> <br> </p>', unsafe_allow_html=True) #st.write("Uploaded noisy audio:")

    col3, col4 = st.columns(2, gap='medium')
    with col3:
        st.audio(uploaded_file)

    with col2:
        with st.spinner(f"Enhancing with {enhancer.split('/')[1]}:"):
            cleaned = requests.post(upload_url,
                                files=file,
                                data=params).json()
        st.success(f"Enhanced with {enhancer.split('/')[1]}:")
        #st.write(f"Audio cleaned with {enhancer}:")

    with col4:
        st.audio(os.path.join(serve_out_url, cleaned['cleaned_file_name']), format="audio/wav")
    col5, col6 = st.columns(2, gap='medium')
    with col5:
        with Image.open(requests.get((os.path.join(serve_in_url, cleaned['cleaned_file_name']+'.jpg')), stream=True).raw) as image:
            st.image(image)
        #st.image(os.path.join(serve_in_url, cleaned['cleaned_file_name'])+'.jpg')
    with col6:
        with Image.open(requests.get((os.path.join(serve_out_url, cleaned['cleaned_file_name']+'.jpg')), stream=True).raw) as image:
            st.image(image)
        #st.image(os.path.join(serve_out_url, cleaned['cleaned_file_name'])+'.jpg')
