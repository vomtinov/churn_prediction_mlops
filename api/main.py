from fastapi import FastAPI
from pydantic import BaseModel
from api.model import predict_churn

app = FastAPI()

class CustomerFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all required features with types

@app.post("/predict/")
def predict(data: CustomerFeatures):
    input_data = data.dict()
    return predict_churn(input_data)
