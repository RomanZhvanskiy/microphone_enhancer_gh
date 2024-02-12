"""
API test
"""

import os
import requests
import streamlit as st

url = "https://micenhancerapi-3t3dih6maa-oa.a.run.app/" # "https://micenhancerapi-3t3dih6maa-oa.a.run.app/", "http://127.0.0.1:8000/"
upload_url = url + 'upload_file'
serve_out_url = url + 'audio_out'

'''
### Test FE for audio enhancers
'''
enhancer = st.radio("Select enhancer:",
                    ["speechbrain/sepformer-dns4-16k-enhancement",
                     "speechbrain/sepformer-wham16k-enhancement",
                     "NOT IMPLEMENTED YET: microphone_enhancer_gh"],
                    index=0,
                    )
st.write(enhancer, " is selected.")
uploaded_file = st.file_uploader("Choose a noisy audio file (.wav):", type='wav')
if uploaded_file is not None:
    st.write("Uploaded noisy audio:")
    st.write(uploaded_file)
    st.audio(uploaded_file)
    file = {'file': uploaded_file}
    params = {'enhancer': enhancer}
    cleaned = requests.post(upload_url,
                            files=file,
                            data=params).json()
    st.write(f"Audio cleaned with {enhancer}:")
    st.write(cleaned)
    st.audio(os.path.join(serve_out_url, cleaned['cleaned_file_name']), format="audio/wav")
