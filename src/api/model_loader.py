from tensorflow.keras.models import load_model #type: ignore

class ModelLoader:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, data):
        return self.model.predict(data)

# Inicializa o modelo globalmente
model_loader = ModelLoader(model_path='models\lstm_stock_prediction_model.h5')
