FROM vllm/vllm-openai:latest

# Install Ministral 3 dependencies FIRST
RUN pip install --no-cache-dir \
    "transformers==5.0.0rc0" \
    "mistral-common>=1.8.6"

# Then install vLLM from source (nightly has Ministral 3 support)
RUN pip install --no-cache-dir \
    vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly