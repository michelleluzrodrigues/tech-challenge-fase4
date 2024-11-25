import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PreProcessor:
    def __init__(self, sequence_length: int, num_features: int):
        """
        Inicializa o pré-processador com os parâmetros esperados pelo modelo.
        :param sequence_length: Número de passos temporais esperados pelo modelo.
        :param num_features: Número de features em cada instante temporal.
        """
        self.sequence_length = sequence_length
        self.num_features = num_features

    def preprocess_input(self, data: list) -> np.ndarray:
        """
        Pré-processa os dados para o formato esperado pelo modelo:
        - Normaliza os dados entre 0 e 1.
        - Adiciona padding ou corta os dados para ajustar ao tamanho esperado.
        - Ajusta para o formato [1, sequence_length, num_features].
        
        :param data: Lista bidimensional com os dados de entrada.
        :return: Dados transformados prontos para o modelo.
        """
        data = np.array(data)

        # Verifica se o número de features está correto
        if data.shape[1] != self.num_features:
            raise ValueError(
                f"Número de features incorreto. Esperado: {self.num_features}, Recebido: {data.shape[1]}"
            )

        # Normaliza os dados entre 0 e 1
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        # Adiciona padding se necessário
        if data.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - data.shape[0], self.num_features))
            data = np.vstack([padding, data])

        # Corta os dados se forem maiores que sequence_length
        if data.shape[0] > self.sequence_length:
            data = data[-self.sequence_length :, :]

        # Ajusta para o formato [1, sequence_length, num_features]
        return data.reshape(1, self.sequence_length, self.num_features)
