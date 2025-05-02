# Dockerfile for vLLM with AMD MI300X optimization
# Based on AMD's ROCm vLLM Docker image

FROM rocm/vllm:rocm6.4.0_mi300x_ubuntu22.04_py3.12_vllm_0.6.6

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    gradio==4.19.* \
    qdrant-client==1.7.* \
    pydantic==2.* \
    fastapi==0.109.* \
    uvicorn==0.27.* \
    prometheus-client==0.19.*

# Copy configuration files
COPY config/vllm_config.yaml /app/config/vllm_config.yaml
COPY scripts/entrypoint.sh /app/entrypoint.sh

# Add warmup scripts
COPY scripts/warmup.py /app/scripts/warmup.py

# Make scripts executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ROCM_VISIBLE_DEVICES=0
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose ports
EXPOSE 8000

# Create volume mount points
VOLUME /data

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]