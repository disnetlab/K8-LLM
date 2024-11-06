# Use LLama to generated prompt that satisfies number of tokens
import requests
import os
import pandas as pd
import json

bearer_token = os.getenv("BEARER_TOKEN")

url = "https://api.deepinfra.com/v1/openai/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {bearer_token}"
}
model = "meta-llama/Meta-Llama-3-8B-Instruct"

def generate_prompt(tokens):
    """
    Generates a concrete prompt with the requested number of tokens in the prompt.
    """
    prompt = f"Write a high-quality and meaningful story about a random topic. Make the story exactly {tokens} tokens. Do not mention the number of tokens."

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_new_tokens": tokens,
        "temperature": 0.7,
        "stream": False
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        generated_text = response_json['choices'][0]['message']['content'].strip()

        return generated_text
    else:
        return f"Error: {response.status_code}, {response.text}"


def collect_prompts(traces):
    """
    Generates prompt and collects them for later use.
    """
    prompts = []  # List to store the generated paragraphs

    for index, row in traces.iterrows():
        request_id = row['request_id']
        arrival_timestamp = row['arrival_timestamp']
        prompt_size = row['prompt_size']
        token_size = row['token_size']

        prompt = generate_prompt(prompt_size)
        prompts.append({
            "request_id": request_id,
            "arrival_timestamp": arrival_timestamp,
            "prompt_size": prompt_size,
            "token_size": token_size,
            "prompt": prompt
        })

    return prompts

if __name__ == "__main__":
    if not os.path.exists("traces"):
        print("Please run generate_traces.py first to generate traces")
    else:
        if not os.path.exists("prompts"):
            os.makedirs("prompts")

        for file in os.listdir("traces"):
            file_path = os.path.join("traces", file)

            # Generate evaluation prompts
            with open(file_path, 'r') as f:
                traces = pd.read_csv(f)
                prompts = collect_prompts(traces)
                filename = os.path.splitext(file)[0]

                with open(f"prompts/{filename}.json", 'w') as f:
                    json.dump(prompts, f, indent=4)
