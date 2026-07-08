import sys
import pickle
import pandas as pd
from model import preprocessing_utils

sys.modules['preprocessing_utils'] = preprocessing_utils  # Add the module to sys.modules so it can be found during unpickling
# Load model safely
with open("C:/Users/jitenshah/OneDrive/Desktop/Email_Spam_Backend/model/spam_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


def predict_message(message:str)->str:
    data=pd.Series([message])
    prediction=model.predict(data)[0]
    if prediction==1:
        return "The message is a spam"
    else:
        return "The message is not a spam"
