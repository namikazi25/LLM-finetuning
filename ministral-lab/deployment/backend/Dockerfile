FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    "transformers==5.0.0rc0" \
    "mistral-common>=1.8.6" \
    torch \
    fastapi \
    uvicorn[standard] \
    accelerate

WORKDIR /app
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]