import numpy as np

class PreProcessor:
    def __init__(self, sequence_length: int, num_features: int):
        self.sequence_length = sequence_length
        self.num_features = num_features

    def transform(self, data: list) -> np.ndarray:
        """
        Prepara os dados no formato esperado pelo modelo:
        - Adiciona padding se necessário.
        - Converte para NumPy e ajusta as dimensões.
        """
        data = np.array(data)

        # Verifica se o número de features está correto
        if data.shape[1] != self.num_features:
            raise ValueError(f"O número de features esperado é {self.num_features}, mas foi fornecido {data.shape[1]}.")

        # Adiciona padding se o comprimento da sequência for menor que o esperado
        if data.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - data.shape[0], self.num_features))
            data = np.vstack([padding, data])

        # Reshape para incluir a dimensão do batch (necessário para predição)
        return data.reshape(1, self.sequence_length, self.num_features)
