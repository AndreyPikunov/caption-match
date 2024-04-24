FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    # Clean up the package list to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py"]
