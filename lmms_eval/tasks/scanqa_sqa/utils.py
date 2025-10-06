import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger


PROMPTS = {
    'scanQA': {
        'template': 'Question: {Question}\nPlease answer in ONE word or a short phrase only.\n',
    },
    'sqa': {
        'template': '{situation}\nQuestion: {Question}\nPlease answer in ONE word or a short phrase only.\n',
    }
}

def get_prompt(dataset_name: str):
    if dataset_name in PROMPTS:
        return PROMPTS[dataset_name]['template']
    else:
        raise ValueError(f"Dataset {dataset_name} not found in PROMPTS")

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def scanqa_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["scene_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    
    question = doc["question"]
    prompt = get_prompt('scanQA')
    return prompt.format(Question=question)


    
