FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in correct order
RUN pip3 install --no-cache-dir \
    "transformers==5.0.0rc0" \
    "mistral-common>=1.8.6"

# Install vLLM nightly (has Ministral 3 + transformers 5 support)
RUN pip3 install --no-cache-dir \
    vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly

WORKDIR /workspace