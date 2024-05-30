from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# API'ye aşağıdaki formatta veri beklediğimizi belirtiyoruz.
# Bu model girdilerimizi uygulayacağımız model
class ModelInput(BaseModel):
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int


BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/ml_model.pkl", "rb") as f:
    heart_attack_model = joblib.load(f)

# API endpoint'i tanımlama
@app.post('/heart_prediction')
def heart_prediction(input_parameters: ModelInput):
    
    # Pydantic modelini sözlük formatına dönüştürme
    input_data_dict = input_parameters.dict()
    
    # Modelin tahmin yapması için gerekli olan veri listesini oluşturma
    input_list = [input_data_dict["sex"], input_data_dict["cp"], 
                  input_data_dict["trtbps"], input_data_dict["chol"], input_data_dict["fbs"], 
                  input_data_dict["restecg"], input_data_dict["thalachh"], input_data_dict["exng"], 
                  input_data_dict["oldpeak"], input_data_dict["slp"], input_data_dict["caa"], 
                  input_data_dict["thall"]]
    
    # Modelin tahmin yapması
    prediction = heart_attack_model.predict([input_list])
    
    # Tahmin sonucuna göre dönüş yapma
    if prediction[0] == 0:
        return 'Bu kişi sağlıklı'
    else:
        return 'Bu kişide kalp krizi riski var!'
