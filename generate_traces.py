# ------------------------------------------------------------------------
# This code is adapted on code originally written in the paper:
#  
# Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, 
# Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient Generative LLM 
# Inference Using Phase Splitting", in Proceedings of the International 
# Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, 
# Argentina, 2024.
#
# Source Code: https://github.com/Mutinifni/splitwise-sim
# Date Accessed: September 2024
# ------------------------------------------------------------------------

import os
from collections import namedtuple
import requests
import numpy as np
import pandas as pd
from scipy import stats

Distributions = namedtuple('Distributions', ['arrival_process',
                                             'prompt_size',
                                             'token_size'])
Distribution = namedtuple('Distribution', ['name', 'params'])

def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":
        return np.ones(size) * params["value"]
    elif distribution == "normal":
        return stats.norm(**params).rvs(size=size)
    elif distribution == "truncnorm":
        return stats.truncnorm(**params).rvs(size=size)
    elif distribution == "randint":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "uniform":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "exponential":
        return stats.expon(**params).rvs(size=size)
    elif distribution == "poisson":
        return stats.poisson(**params).rvs(size=size)
    elif distribution == "trace":
        df = pd.read_csv(params["filename"])
        return df[params["column"]].sample(size, replace=True).values
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

def generate_trace(max_requests, distributions, end_time=None):
    """
    Generate a trace of requests based on the given distributions.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)
    arrival_timestamps = np.cumsum(arrival_timestamps)

    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)
    prompt_sizes = map(int, prompt_sizes)
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)
    token_sizes = map(int, token_sizes)

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,
        "arrival_timestamp": arrival_timestamps,
        "prompt_size": prompt_sizes,
        "token_size": token_sizes,
    })

    if end_time is not None:
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]

    return trace_df


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),
    )
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df



def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)
        trace_filename = trace_filename_template.format(request_rate)
        trace_df.to_csv(trace_filename, index=False)


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)
    
def download_file(url, filename):
    """
    Download a file from the given URL.
    """
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


def download_azure_llm_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"

    if not os.path.exists("data/conv_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_conv.csv"
        download_file(url, "data/conv_distributions.csv")
        print("Downloaded conv traces")

if __name__ == "__main__":
    # download prompt and token size distributions
    download_azure_llm_traces()

    generate_conv_traces(
        max_requests=10000,
        end_time=5, # in seconds
        request_rates=list(range(5, 21, 5)), # requests per second
        conv_distributions_file="data/conv_distributions.csv")