# Recommendation Systems using Python

## Folder Strcture:

├── Main Files:
    .gitattributes
    .gitignore
    CNN.py
    HMM.py
    KNN.py
    LSTM.py
    README.md
    ReinforcementLearning.py
├───Checkpoints
        ActorModel.pth
        CriticModel.pth
        HMM.pth
        LSTM.pth
        MainModel.pth
        Model1.pth
├───DatasetsPreprocess
        HMM_Preprocess.py
        KNN_Preprocess.py
        LSTM_Preprocess.py
        RL_Preprocess.py
├───Graphs
        HMM_AVG_Loss.png
        HMM_Loss.png
        HMM_VS_LSTM_Loss.png
        LSTM_AVG_Loss.png
        LSTM_Loss.png
        RL_Loss.png
├───Models
        CNN_Model.py
        HMM_Model.py
        LSTM_Model.py
        ReinforcementLearning_Model.py
└───Results
        HMM_Results.csv
        HMM_Results_Sample.png
        KNN_Results.csv
        KNN_Results_Sample.png
        LSTM_Results.csv
        LSTM_Results_Sample.png
        Query.png
        QueryResult.png
        RL_Results_Sample.csv
        RL_Results_Sample.png

## How to run the code:

1. Install Python in Local Computer. Preferably latest version. We are using 3.9.11
2. Install Tensorflow, Pytorch, Sklearn, Scikit, Matplotlib, Pandas, and Numpy using Pip or Conda package Installer
3. Set path in the environment variables
4. Clone the repository into a folder
5. The files CNN.py, HMM.py, KNN.py, LSTM.py and ReinforcementLearning.py are the main file of the project
6. Dataset can be found at kaggle (H&M Personalized Fashion Recommendations)
7. We have uploaded our results in Results folder. The performance of the models can be seen in Graphs folder 

## pip Commands:

-    pip install tensorflow
-    pip install torch
-    pip install sklearn
-    pip install pandas
-    pip install numpy
-    pip install matplotlib

## Datase Used:
Link: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

## Note:
We have tried to upload the CSV Files and Checkpoint files of the models but they exceed the space alloted by GitHub. So, uploaded those files in a Drive
Link: https://drive.google.com/drive/folders/1chbjW1SgBpT-Z8u61A7q1LM7e4o699gz?usp=share_link

