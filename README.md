# Tech Challenge Fase 4 - FIAP

## Introdução

Este projeto foi desenvolvido como parte da fase 4 do Tech Challenge, na FIAP. O objetivo desta etapa é aplicar conhecimentos avançados de machine learning e deep learning para resolver um problema de previsão de séries temporais.

O projeto consiste na construção de um modelo de **Deep Learning** utilizando redes **LSTM (Long Short-Term Memory)**, que são especialmente eficazes para trabalhar com dados sequenciais. Nosso modelo é projetado para prever o preço das ações da Coca-Cola com base em dados históricos de fechamento, obtidos diretamente da plataforma **Yahoo! Finance**.

---

## Como desenvolver o projeto localmente

O projeto foi criado utilizando o gerenciador de pacotes **Poetry**, o que facilita a instalação de dependências e o ambiente de desenvolvimento. Para configurar o projeto localmente, siga os passos abaixo:

1. **Instale o Poetry**
   Certifique-se de que o Poetry está instalado na sua máquina. Você pode instalá-lo seguindo as instruções em [Poetry Docs](https://python-poetry.org/docs/) ou também com `pip install poetry`.

2. **Clone o repositório**

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

3. **Instale as dependências**

   ```bash
   poetry install
   ```

4. **Ative o ambiente virtual**

   ```bash
   poetry shell
   ```

Após essas etapas, o ambiente de desenvolvimento estará configurado para trabalhar no projeto.

---

## Como usar o projeto com Docker

Se preferir rodar o projeto utilizando **Docker**, você pode seguir as etapas abaixo:

1. **Build da imagem Docker**
   Execute o comando para criar a imagem:

   ```bash
   docker build -t tech-challenge-4 .
   ```

2. **Inicie o container**
   Rode o seguinte comando para iniciar a aplicação:

   ```bash
   docker run -p 8000:8000 tech-challenge-4
   ```

Após isso, o projeto estará rodando no endereço `http://localhost:8000`.

---

## Como utilizar a API

A API oferece uma rota principal chamada `/predict` para fazer previsões com o modelo. Além disso, você pode acessar a interface interativa da API em `http://localhost:8000/docs/`, que é gerada automaticamente pelo **FastAPI**.

### Estrutura da Requisição

A rota `/predict` aceita requisições **POST** com os seguintes parâmetros:

- `days` (inteiro): Número de dias que o modelo deve prever.
- `data` (array): Dados históricos de fechamento, organizados em listas com 6 valores cada (exemplo abaixo).

### Exemplo de Requisição

Você pode enviar a requisição com o **cURL**:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "days": 30,
  "data": [
      [132.7, 133.1, 134.2, 135.0, 134.8, 134.0],
      [134.5, 135.3, 136.1, 135.9, 135.6, 136.0],
      [136.7, 137.2, 137.0, 132.7, 133.1, 134.2],
      [135.0, 134.8, 134.0, 134.5, 135.3, 136.1],
      [135.9, 135.6, 136.0, 136.7, 137.2, 137.0],
      [132.7, 133.1, 134.2, 135.0, 134.8, 134.0],
      [134.5, 135.3, 136.1, 135.9, 135.6, 136.0],
      [136.7, 137.2, 137.0, 132.7, 133.1, 134.2],
      [135.0, 134.8, 134.0, 134.5, 135.3, 136.1],
      [135.9, 135.6, 136.0, 136.7, 137.2, 137.0]
  ]
}'
```

### Resposta Esperada

A API retornará as previsões para os próximos `days` informados. As previsões serão baseadas nos dados históricos fornecidos em `data`.

---
