# Use uma imagem base leve do Python
FROM python:3.12-slim

# Configure o diretório de trabalho
WORKDIR /app

# Copie os arquivos do Poetry para o container
COPY pyproject.toml poetry.lock ./

# Instale as dependências sem criar um virtualenv
RUN pip install poetry==1.8.4 && \
    poetry config virtualenvs.create false && \
    poetry install --without dev --no-root --no-directory

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "fastapi", "run", "src/api/app.py", "--host", "0.0.0.0", "--port", "8000"]
