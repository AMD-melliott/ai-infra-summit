# Model Serving Swiss Army Knife: Multi-Modal AI Inference on AMD Instinct

## Overview

This repository contains the files needed to set up a demo model serving platform on AMD Instinct GPUs. By splitting the GPU into smaller virtual parts, you can efficiently run different models simultaneously and access them all through a single API.

This demo was originally presented at [AI Infra Summit 3](https://aiinfra.live) on May 3, 2025.

## Key Features

- **Multi-Model Deployment**: Deploy multiple models of different sizes and architectures simultaneously
- **GPU Partitioning**: Efficient utilization of AMD Instinct GPUs CPX partition mode (8 Logical GPUs per Physical GPU)
- **Unified API**: Single OpenAI-compatible endpoint for accessing all models through LiteLLM proxy
- **Multiple Inference Engines**: Support for vLLM and SGLang to optimize for different model architectures
- **Real-time Monitoring**: Status dashboard showing deployed models and GPU VRAM utilization
- **Interactive UI**: Web interface for testing models and generating loads
- **Load Testing**: Built-in tools to simulate workloads and test scaling

**Many thanks** to [Vultr](https://www.vultr.com/) for providing an AMD Instinct MI325X bare metal instance to build and host this demo. Find more information about Vultr's Instinct-based offerings here: [https://www.vultr.com/products/cloud-gpu/amd-mi325x-mi300x/](https://www.vultr.com/products/cloud-gpu/amd-mi325x-mi300x/)

### Screenshots

<div align="center">

![Metrics Dashboard](images/dashboard.png)
*Metrics Dashboard*

</div>
<p>
<div align="center">

![Load Generator UI](images/load-gen.png)
*Demo Load Generator UI*

</div>

## Prerequisites

* **AMD Instinct™ GPU**: You'll need access to an AMD Instinct GPU with the [drivers installed](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
* **Docker**: Docker and Docker Compose to run the containerized services.
* **Hugging Face Account**: A Hugging Face token if you want to use any private models.

> **Note:** This demo uses `network_mode: host` for simplicity, which means the containers share your server's network. This makes things easier but isn't recommended for production environments without additional security measures.

## Architecture

The demo is made up of these components:

* **Model Servers**: Multiple vLLM and SGLang servers, each running different AI models
* **Central API**: A LiteLLM server that routes your requests to the right model
* **Web Interface**: A simple UI for interacting with deployed models
* **Monitoring**: A dashboard showing what's running and how resources are being used
* **Load Testing**: Tools to simulate many users and test performance

### Logical Diagram

<div align="center">

![Load Generator UI](images/swiss-army-knife.png)
*Logical Diagram*

</div>


## Default Models

The following models are pre-configured:

| Model | Type | Server | GPU Partitions |
|-------|------|--------|----------------|
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | LLM | vLLM | 0-3 |
| [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) | LLM | vLLM | 4-7 |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | LLM | vLLM | 8-11 |
| [mistralai/Mistral-Nemo-Instruct-FP8-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-FP8-2407) | LLM | vLLM | 12-15 |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | Embedding | vLLM | 16-19 |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reranker | vLLM | 20-23 |
| [google/gemma-3-4B-it](https://huggingface.co/google/gemma-3-4B-it) | LLM | vLLM | 32-35 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | LLM | vLLM | 36-39 |
| [CohereLabs/c4ai-command-r7b-12-2024](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024) | LLM | vLLM | 40-43 |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | LLM | SGLang | 24-27 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | LLM | SGLang | 28-31 |

## Port Usage

This demo uses the following TCP ports:

| Service | Port Range |
|---------|------------|
| vLLM Servers | 8000-8043 |
| SGLang Servers | 9101-9108 |
| LiteLLM Proxy | 4000 |
| Open WebUI | 8081 |
| Metrics Aggregator | 5001 |
| Load Generator | 5002 |

## GPU Partitioning Setup

This demo uses a feature of AMD GPUs that lets you split one physical GPU into multiple virtual GPUs. To run several models at once, we need to set the GPU to "CPX Partition" mode.

### Understanding SPX and CPX Modes

AMD Instinct GPUs support different compute partition modes that affect how the GPU presents itself to applications:

- **SPX (Single Partition X-celerator)**: The default mode where all compute chiplets (XCDs) appear as a single logical GPU. This provides a unified view but offers less control over workload placement.

- **CPX (Core Partitioned X-celerator)**: Each compute chiplet appears as a separate logical GPU (8 separate GPUs per MI300X/MI325X). This gives explicit control over which chiplet runs which workload, allowing multiple models to run simultaneously on separate partitions.

### Checking and Setting Partition Mode

To see how your GPU is currently set up, check current partition mode for GPU 0:

```bash
sudo amd-smi partition -g 0
```

If it shows "SPX" mode, change it to "CPX" mode with:

```bash
sudo amd-smi set -C cpx
```

When you're done with the demo and want to switch back:

```bash
sudo amd-smi set -C spx
```

For more details, check out the [AMD ROCm documentation](https://rocm.docs.amd.com/) and this [blog post about Compute and Memory Modes](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html).

## Getting Started

### 1. Configure Environment

Create your configuration file by copying the example:

```bash
cp env.example .env
```

Open the `.env` file and update it with:

- Where you want to store your models
- Where to cache data
- Your Hugging Face token (if you need private models)

### 2. Download Models

Make sure all the models listed in `models.yaml` are downloaded to your model storage directory.

### 3. Start Services

Launch everything with `docker compose`:

```bash
docker compose up -d
```

This starts all the needed services in the background. Note that it can take several minutes for all inference services to fully start and load the models. Once all services are running, use the LiteLLM `/health` endpoint to validate that inference servers are working.

```bash
curl -sS http://localhost:4000/health | jq
```

```json
{
  "healthy_endpoints": [
    {
      "model": "openai/Qwen/Qwen3-4B",
      "api_base": "http://localhost:8000/v1"
    },
    {
      "model": "openai/Qwen/Qwen3-4B",
      "api_base": "http://localhost:8001/v1"
    },
    {
      "model": "openai/Qwen/Qwen3-4B",
      "api_base": "http://localhost:8002/v1"
    },
[output truncated]
    {
      "model": "openai/CohereLabs/c4ai-command-r7b-12-2024",
      "api_base": "http://localhost:8041/v1"
    },
    {
      "model": "openai/CohereLabs/c4ai-command-r7b-12-2024",
      "api_base": "http://localhost:8042/v1"
    },
    {
      "model": "openai/CohereLabs/c4ai-command-r7b-12-2024",
      "api_base": "http://localhost:8043/v1"
    }    
  ],
  "healthy_count": 44,
  "unhealthy_count": 0
}                    
```

### 4. Access Interfaces

- **Status Dashboard**: http://localhost:8080
- **Open WebUI**: http://localhost:8081
- **Load Generator**: http://localhost:8080/load-generator/
- **LiteLLM API**: http://localhost:4000

### 5. Accessing Remotely via SSH Tunneling

Since this demo runs on enterprse server hardware, you'll likely be running it on a remote machine or cloud provider. To view the web interfaces on your local computer, you can create a secure connection using SSH:

```bash
# Connect to your remote server and make the web interfaces available locally
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 -L 4000:localhost:4000 username@remote-server
```

After connecting, simply open your web browser and go to the local URLs listed above (like http://localhost:8080). Your browser will connect to the remote server through the secure tunnel you created.

For convenience, you can save these settings in your SSH config file (` ~/.ssh/config `):

```bash
# Host ai-demo
#   HostName remote-server-ip-or-name
#   User username
#   LocalForward 8080 localhost:8080
#   LocalForward 8081 localhost:8081
#   LocalForward 4000 localhost:4000
```

### 6. Monitor Logs

```bash
docker compose logs -f <service_name>  # e.g., docker compose logs -f proxy
```

### 7. Stop Services

```bash
docker compose down
```

## Customization

### Adding New Models

Want to add your own models? It's easy:

1. Download the model to your models directory
2. Add your model's details to the `models.yaml` file
3. Edit `docker-compose.vllm.yaml` and/or `docker-compose.sglang.yaml` as needed
4. Bring up the new models with `docker compose up -d`
4. Restart the proxy with `docker compose restart proxy`

## Questions?

Please open an [issue](https://github.com/AMD-melliott/ai-infra-summit/issues) in this repo with any questions.