# K8-LLM
## Introduction
This repository contains a simple LLM workload generator to benchmark LLM performance (including its Time To First Token, End-to-End Request Latency, Inter-Token Latency) on a Kubernetes cluster.

## Setup
```python
# Clone this repository
git clone https://github.com/disnetlab/K8-LLM.git
cd K8-LLM

# Create and activate new Python environment
python -m venv myenv
source myenv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```
**NOTE**: The code has only been tested with Python 3.11. However, it will likely also work with other Python versions.

## Prepare Kubernetes Cluster
You can set up your Kubernetes cluster by following the [Kubenetes Cluster Setup Tutorial](https://github.com/techiescamp/kubeadm-scripts/tree/main?tab=readme-ov-file) or use other preferred methods for configuring your cluster.

## Prepare Server
To run this generator, you need an Ollama LLM server running on a Kubernetes node. The ```ollama-depl.yaml``` file in the ```ollama``` directory contains the deployment configuration needed for setting up the server within a node. <br />

If you considering a different LLM model, modify the model name in the line ```command: ["/bin/sh", "-c", "ollama pull qwen2:0.5b"]``` within the deployment file. 
Make sure the model name follows Ollama's naming standards. For more available models, visit the [Ollama Models](https://ollama.com/library). <br />

Environment variables for the Ollama server in the file can be adjusted as needed.

- `OLLAMA_KEEP_ALIVE`: Duration model stays loaded in memory. 
- `OLLAMA_MAX_QUEUE`: Maximum queued requests before rejection.

For more details on configuration options, refer to the 
[Ollama Environment Variables](https://github.com/ollama/ollama/blob/9d71bcc3e2a97c8e62d758450f43aa212346410e/docs/faq.md) <br />

## Prepare Traces
The ```generate_trace.py``` script automatically downloads the production traces and uses the corresponding prompt and response size distributions to generate request traces with different request rates. Modify and run ```generate_trace.py``` with desired request rates and other parameters. <br />

Trace distribution can be visualised in the ```plot.ipynb``` in the ```notebook``` directory.

## Prepare Evaluation Datasets
The ```generate_eval.py``` script generates the evaluation dataset used for the generator. A commercial platform’s LLM API [DeepInfra](https://deepinfra.com/) is used to generate concrete prompts. <br /> 

To get started, follow the instructions on the platform to set up an inference API, specifying the ```meta-llama/Meta-Llama-3-8B-Instruct``` model. Make sure to set the provided Bearer token in your environment variable accordingly. 

```
On Linux/Mac: export BEARER_TOKEN="your_token_here"
On Windows: set BEARER_TOKEN=your_token_here
```

Modify the ```generate_eval.py``` if you consider using different model/platform.

## Run Generator
Modify and run ```generate_trace.py``` with desired model, request rate, cluster nodes IPs and request payload. 

For more details on configuration options for the payload, refer to the [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md).


## Attribution
The trace data utilised in this experiment originates from the following publication:
> Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. "[**Splitwise: Efficient generative LLM inference using phase splitting**](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/)", in Proceedings of the International Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024. 



