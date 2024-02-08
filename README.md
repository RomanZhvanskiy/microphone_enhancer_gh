FOLDER STRUCTURE                                                                                  <br />
=======================                                                                           <br />
├── Back_end                                                                                      <br />
│   ├── api                                                                                       <br />
│   │   ├── api_func.py                                                                           <br />
│   │   ├── enhancer_api.py                                                                       <br />
│   ├── audio_preprocessing                                                                       <br />
│   │   ├── preprocessing.py         #conversion of waveforms to/from SG, degrade quality, etc    <br />
│   ├── hugging_models                                                                            <br />
│   │   ├── hugrestore.py                                                                         <br />
│   ├── image_metrics                                                                             <br />
│   │   ├── img_metrics.py           #image quality metrics                                       <br />
│   ├── interface                                                                                 <br />
│   │   ├── audioenhancer_local.py   #this is where the functions are to be called by API         <br />
│   ├── ml_logic                                                                                  <br />
│   │   ├── model.py                 #NN model is here                                            <br />
│   ├── params.py                    #parameters (mostly folder names) here                       <br />
│                                                                                                 <br />
├── Data                                                                                          <br />
│   ├── audio_data                                                                                <br />
│   │   ├── audio_in                  #this is where bad quality input comes in                   <br />
│   │   └── audio_out                 #this is where good quality output comes out                <br />
│   ├── postprocessed_training_data   #preprocessed data for training model                       <br />
│   │   ├── degraded_test_sg.sg                                                                   <br />
│   │   ├── degraded_train_sg.sg                                                                  <br />
│   │   ├── test_sg.sg                                                                            <br />
│   │   └── train_sg.sg                                                                           <br />
│   ├── pretrained_models             #where we save models                                       <br />
│   └── raw_data                      #data for training model                                    <br />
│       └── VCTK-Corpus                                                                           <br />
├── Front_end                                                                                     <br />
│   └── test_api.py                   #web site                                                   <br />
├── jupyter_books                                                                                 <br />
│   ├── Copy of Speech_enhancement_6-checkpoint.ipynb                                             <br />
│   └── hugging_conversions_test.ipynb                                                            <br />
├── Makefile                                                                                      <br />
├── README.md                                                                                     <br />
├── requirements.txt                                                                              <br />
├── setup.py                                                                                      <br />
├── test_hugging_models.py                                                                        <br />
└── Untitled.ipynb                                                                                <br />
