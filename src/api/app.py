from fastapi import FastAPI, Query  # Importa Query para parâmetros opcionais
from pydantic import BaseModel
from typing import List
from model_loader import model_loader
from pre_processor import PreProcessor
import numpy as np

app = FastAPI()

class StockInput(BaseModel):
    data: List[List[float]]  # Matriz de dados de entrada


# Inicialize o pré-processador com os parâmetros esperados pelo modelo
pre_processor = PreProcessor(sequence_length=60, num_features=6)

@app.post("/predict")
def predict(input_data: StockInput, days: int = Query(1, ge=1)):
    try:
        # Pré-processa os dados para o formato esperado
        current_data = pre_processor.transform(input_data.data)

        # Itera para prever múltiplos dias
        results = []
        for _ in range(days):
            prediction = model_loader.predict(current_data)
            results.append(prediction.tolist())

            # Se o modelo for autoregressivo, atualize os dados para a próxima predição
            # current_data = update_with_prediction(current_data, prediction)

        return {"predictions": results}

    except ValueError as e:
        return {"error": str(e)}
