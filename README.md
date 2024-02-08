FOLDER STRUCTURE                                                                                  #
=======================                                                                           #
├── Back_end                                                                                      #
│   ├── api                                                                                       #
│   │   ├── api_func.py                                                                           #
│   │   ├── enhancer_api.py                                                                       #
│   ├── audio_preprocessing                                                                       #
│   │   ├── preprocessing.py         #conversion of waveforms to/from SG, degrade quality, etc    #
│   ├── hugging_models                                                                            #
│   │   ├── hugrestore.py                                                                         #
│   ├── image_metrics                                                                             #
│   │   ├── img_metrics.py           #image quality metrics                                       #
│   ├── interface                                                                                 #
│   │   ├── audioenhancer_local.py   #this is where the functions are to be called by API         #
│   ├── ml_logic                                                                                  #
│   │   ├── model.py                 #NN model is here                                            #
│   ├── params.py                    #parameters (mostly folder names) here                       #
│                                                                                                 #
├── Data                                                                                          #
│   ├── audio_data                                                                                #
│   │   ├── audio_in                  #this is where bad quality input comes in                   #
│   │   └── audio_out                 #this is where good quality output comes out                #
│   ├── postprocessed_training_data   #preprocessed data for training model                       #
│   │   ├── degraded_test_sg.sg                                                                   #
│   │   ├── degraded_train_sg.sg                                                                  #
│   │   ├── test_sg.sg                                                                            #
│   │   └── train_sg.sg                                                                           #
│   ├── pretrained_models             #where we save models                                       #
│   └── raw_data                      #data for training model                                    #
│       └── VCTK-Corpus                                                                           #
├── Front_end                                                                                     #
│   └── test_api.py                   #web site                                                   #
├── jupyter_books                                                                                 #
│   ├── Copy of Speech_enhancement_6-checkpoint.ipynb                                             #
│   └── hugging_conversions_test.ipynb                                                            #
├── Makefile                                                                                      #
├── README.md                                                                                     #
├── requirements.txt                                                                              #
├── setup.py                                                                                      #
├── test_hugging_models.py                                                                        #
└── Untitled.ipynb                                                                                #
