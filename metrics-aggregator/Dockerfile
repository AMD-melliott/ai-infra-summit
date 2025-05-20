ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:6.4-complete
FROM ${BASE_IMAGE}

# Set the working directory in the container
WORKDIR /app

# Install Python and virtual environment support
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv python3-full python3-psutil python3-requests python3-flask python3-werkzeug python3-urllib3 && \
    apt-get clean

# Copy all application files
COPY . .

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV ROCM_SMI_VRAM_OVERRIDE=true
ENV FLASK_APP=metrics_aggregator.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Expose port for API
EXPOSE 5001

# Set entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Use entrypoint script to start the appropriate metrics aggregator
ENTRYPOINT ["/app/entrypoint.sh"] 