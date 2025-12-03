FROM vllm/vllm-openai:latest

# Remove conflicting default libraries
RUN pip uninstall -y transformers vllm

# Install the specific RC versions required by Ministral 3 docs
RUN pip install --no-cache-dir \
    "transformers==5.0.0rc0" \
    "mistral-common>=1.8.6"

# Install vLLM Nightly (Required for Ministral 3 architecture)
RUN pip install -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
