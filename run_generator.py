import asyncio
import aiohttp
import json
import time
import itertools
import logging
from aiohttp import ClientTimeout
import os

PROMPT_RATE = "5" # Define the desired evaluation rate
PROMPT_PATH = f"./prompts/rr_conv_{PROMPT_RATE}.json" 
MODEL = "qwen2:0.5b" # Define the desised model
URL_PATH = "/api/generate"

# List of worker node IPs and the NodePort
worker_nodes = [
    "http://172.26.129.239:30000",  # Worker Node 1 
    "http://172.26.133.188:30000"   # Worker Node 2 
]

ttfts = []  
e2es = []   
lock = asyncio.Lock()  

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

node_iterator = itertools.cycle(worker_nodes)

async def send_request(session, id, prompt, token_size):
    """
    Send a request to the model server.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "num_predict": token_size, # Upper bound for response tokens
            "num_ctx": 8192,
            "num_thread": 1,
        },
        "stream": True  # Enable streaming
    }

    node_ip = next(node_iterator) # Round robin selection of worker nodes

    while True:
        try:
            start_time = time.time()
            async with session.post(f"{node_ip}{URL_PATH}", json=payload) as response:
                if response.status != 200:
                    # logging.error(f"Request {id} failed with status code {response.status}. Retrying...")
                    await asyncio.sleep(1)  # Wait a bit before retrying
                    continue  # Retry the request

                ttft = None
                e2e = None

                # Iterate over the streaming response
                async for line in response.content:
                    if not line:
                        continue  # Skip empty lines
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line:
                        continue  # Skip lines that are just whitespace
                    
                    try:
                        response_json = json.loads(decoded_line)
                    except json.JSONDecodeError:
                        logging.error(f"Request {id}: Failed to decode JSON line: {decoded_line}")
                        continue
                    
                    if 'response' in response_json:
                        # Capture TTFT at the first token
                        if ttft is None:
                            ttft = time.time() - start_time
                            async with lock:
                                ttfts.append({"id": id, "ttft": ttft})
                            logging.info(f"TTFT for request {id}: {ttft:.4f} seconds")
                        
                        # Check if the response is done
                        if response_json.get('done', False):
                            break  

                # Capture total E2E after the stream is complete
                e2e = time.time() - start_time
                async with lock:
                    e2es.append({"id": id, "e2e": e2e, "token_size": token_size})
                logging.info(f"RT for request {id}: {e2e:.4f} seconds")
                return  # Exit after successful request

        except asyncio.CancelledError:
            logging.error(f"Request {id} was cancelled.")
            raise
        except asyncio.TimeoutError:
            logging.error(f"Request {id} timed out. Retrying...")
        except Exception as e:
            logging.error(f"Request {id} encountered an error: {e}. Retrying...")

        await asyncio.sleep(1)  # Wait a bit before retrying


async def benchmark():
    """
    Run the benchmark.
    """
    # Load prompts
    try:
        with open(PROMPT_PATH, "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"The {PROMPT_RATE} does not exist.")

    if not prompts:
        raise ValueError("Prompt list is empty. Please check the input file.")

    prompt_iterator = itertools.cycle(prompts)

    timeout = ClientTimeout(
        total=None,         
        connect=None,       
        sock_read=None,     
        sock_connect=None   
    )
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        previous_timestamp = None
        for _ in range(len(prompts)):
            prompt_data = next(prompt_iterator)
            token_size = prompt_data.get('token_size')
            prompt = prompt_data.get('prompt')
            id = prompt_data.get('request_id', f"req_{_+1}")  
            arrival_timestamp = float(prompt_data.get('arrival_timestamp', 0))
            
            if previous_timestamp is not None:
                time_diff = arrival_timestamp - previous_timestamp
                if time_diff > 0:
                    await asyncio.sleep(time_diff)
                else:
                    logging.warning(f"Non-positive time difference encountered for request {id}.")
                    return
            
            task = asyncio.create_task(send_request(
                session, 
                id,
                f"Summarize the following paragraph in exactly {token_size} tokens:\n\n{prompt}",
                token_size
            ))
            tasks.append(task)
            previous_timestamp = arrival_timestamp
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Task resulted in an error: {result}")
        
        logging.info(f"Benchmark completed. TTFTs: {ttfts}")
        logging.info(f"Benchmark completed. E2Es: {e2es}")

if __name__ == "__main__":
    if not os.path.exists("benchmark"):
        os.makedirs("benchmark")

    asyncio.run(benchmark())

    # Output metrics collected
    metrics = []
    for e2e_entry in e2es:
        req_id = e2e_entry["id"]
        e2e = e2e_entry["e2e"]
        token_size = e2e_entry["token_size"]

        # Find corresponding TTFT
        ttft_entry = next((item for item in ttfts if item["id"] == req_id), None)
        ttft = ttft_entry["ttft"] if ttft_entry else None

        metrics.append({
            "id": req_id,
            "ttft": ttft,
            "e2e": e2e,
            "itl": (e2e - ttft) / (token_size - 1) # Calculate ITL
        })
    
    output = f"benchmark/rr_conv_{PROMPT_RATE}.json"
    
    with open(output, "w") as outfile:
        json.dump(metrics, outfile, indent=4)
    
    logging.info(f"Metrics saved to {output}")


