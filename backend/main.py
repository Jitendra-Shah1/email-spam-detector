from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model import preprocessing_utils
from Schema import userinput
from model.predict import predict_message

app = FastAPI()

@app.get('/')
def home():
    return {"message": "Welcome to the Spam Detection API"}

@app.get('/health_check')
def health_check():
    return {"status_code": "Ok"}

@app.post('/predict')
def predict(message: userinput.Message):
    msg = message.message
    try:
        prediction=predict_message(msg)
        return JSONResponse(status_code=200,content={'response':prediction})
    except Exception as e:
        return JSONResponse(status_code=500,content={'error':str(e)})
