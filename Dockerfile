FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    # Clean up the package list to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "scripts/app.py"]
