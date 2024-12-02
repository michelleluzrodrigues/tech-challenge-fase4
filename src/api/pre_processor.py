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
        np_data = np.array(data)

        # Verifica se o número de features está correto
        if np_data.shape[1] != self.num_features:
            raise ValueError(
                f'Número de features incorreto. Esperado: {self.num_features},\
                Recebido: {np_data.shape[1]}'
            )

        scaler = MinMaxScaler()
        np_data = scaler.fit_transform(np_data)

        if np_data.shape[0] < self.sequence_length:
            padding = np.zeros(
                (self.sequence_length - np_data.shape[0], self.num_features)
            )
            np_data = np.vstack([padding, np_data])

        # Corta os dados se forem maiores que sequence_length
        if np_data.shape[0] > self.sequence_length:
            np_data = np_data[-self.sequence_length :, :]

        return np_data.reshape(1, self.sequence_length, self.num_features)
