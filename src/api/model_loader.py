import os

from keras.api.models import Model, load_model


class ModelLoader:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, data):
        return Model(self.model).predict(data)


model_path = os.path.join('models', 'lstm_stock_prediction_model.h5')
model_loader = ModelLoader(model_path=model_path)
