from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from model_loader import model_loader
from pre_processor import PreProcessor  # Import da classe de pré-processamento

app = FastAPI()

# Inicializa o pré-processador com os parâmetros do modelo
pre_processor = PreProcessor(sequence_length=60, num_features=6)

class StockInput(BaseModel):
    data: List[List[float]]  # Entrada esperada como lista de listas (bidimensional)

@app.post("/predict")
def predict(input_data: StockInput, days: int = Query(1, ge=1)):
    """
    Faz a predição com base nos dados de entrada e no número de dias fornecido.
    """
    try:
        # Pré-processa os dados para o formato esperado pelo modelo
        current_data = pre_processor.preprocess_input(input_data.data)

        # Itera para prever múltiplos dias (se solicitado)
        results = []
        for _ in range(days):
            prediction = model_loader.predict(current_data)
            results.append(prediction.tolist())

            # Atualiza os dados para incluir a predição (se o modelo for autoregressivo)
            # Exemplo: current_data = update_with_prediction(current_data, prediction)

        return {"predictions": results}

    except ValueError as e:
        return {"error": str(e)}
