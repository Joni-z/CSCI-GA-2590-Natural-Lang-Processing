import json
import collections
import argparse
import random
import numpy as np
import requests
import re
import os

def your_netid():
    YOUR_NET_ID = 'zz5070'
    return YOUR_NET_ID

def your_hf_token():
    YOUR_HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
    return YOUR_HF_TOKEN


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    prefix = "Add and reply with digits only.\n1234567+1234567=2469134\n"
    suffix = "="

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys. 
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50, # max_tokens must be >= 50 because we don't always have prior on output length
        'temperature': 0.0,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1,
        'stop': []}
    
    return config


def your_pre_processing(s):
    left, right = s.split("+")
    width = max(len(left), len(right))
    return (
        "Compute the exact sum. Return only the final integer with no words.\n"
        f"{left.rjust(width)}\n"
        f"+{right.rjust(width)}\n"
    )

    
def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    match = re.search(r"\d+", output_string)
    return int(match.group(0)) if match else 0
