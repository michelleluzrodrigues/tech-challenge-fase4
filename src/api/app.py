from typing import List

from fastapi import FastAPI, Query
from model_loader import model_loader
from pre_processor import PreProcessor
from pydantic import BaseModel

app = FastAPI()

pre_processor = PreProcessor(sequence_length=60, num_features=6)


class StockInput(BaseModel):
    data: List[List[float]]


@app.post('/predict')
def predict(input_data: StockInput, days: int = Query(1, ge=1)):
    """
    Faz a predição com base nos dados de entrada e no número de dias fornecido.
    """
    try:
        current_data, scaler = pre_processor.preprocess_input(input_data.data)

        results = []
        for _ in range(days):
            prediction = model_loader.predict(current_data)
            results.append(scaler.inverse_transform(prediction).tolist())

        return {'predictions': results}

    except ValueError as e:
        return {'error': str(e)}
