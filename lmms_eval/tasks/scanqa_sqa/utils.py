import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
import re
from loguru import logger as eval_logger

try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
except ImportError:
    eval_logger.debug("pycocoevalcap not installed. Please install pycocoevalcap to use this module. You can install it by running 'pip install pycocoevalcap'")



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

def scanqa_sqa_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["scene_id"] + ".mp4"
    video_path = os.path.join(cache_dir, 'videos', video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    
    question = doc["question"]
    prompt = get_prompt('scanQA')
    return prompt.format(Question=question)


def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    
    question = doc["question"]
    situation = doc["situation"]
    prompt = get_prompt('sqa')
    return prompt.format(situation=situation, Question=question)

def scanqa_process_results(doc, results):
    pred = results[0]
    doc['prediction'] = pred
    return {"scanqa": doc}


def _normalize_for_em(s: str) -> str:
  
    return (s or "").strip().lower()

def _em1(pred: str, refs) -> float:
    p = _normalize_for_em(pred)
    for r in refs or []:
        if p == _normalize_for_em(r):
            return 1.0
    return 0.0


def scanqa_aggregate_results(results):
    
    gts, res = {}, {}
    em_scores = []

  
    for idx, doc in enumerate(results):
        qid = doc.get("question_id")
        pred = (doc.get("prediction")).strip()
        refs = doc.get("answers")

    
        em_scores.append(_em1(pred, refs))

        
        gts[qid] = [{"caption": r} for r in refs]
        res[qid] = [{"caption": pred}]

    out = {"EM1": float(np.mean(em_scores) if em_scores else 0.0)}

    
    tok = PTBTokenizer()
    gts_tok, res_tok = tok.tokenize(gts), tok.tokenize(res)

    
    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts_tok, res_tok)
    for i in range(4):
        out[f"BLEU-{i+1}"] = float(bleu_scores[i])

    
    try:
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts_tok, res_tok)
        out["METEOR"] = float(meteor_score)
    except Exception:
        out["METEOR"] = 0.0  # Java 依赖未安装时防御性返回

    
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(gts_tok, res_tok)
    out["ROUGE_L"] = float(rouge_score)

    return out

def clean_single_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = str(ans)
    s = s.lower()
    s = re.sub(r'[ ]+$', '', s)          
    s = re.sub(r'^[ ]+', '', s)          
    s = re.sub(r' {2,}', ' ', s)         

    s = re.sub(r'\.[ ]{2,}', '. ', s)
    s = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", '', s)

    s = re.sub(r'ç', 'c', s)
    s = re.sub(r'’', "'", s)
    s = re.sub(r'\bletf\b', 'left', s)
    s = re.sub(r'\blet\b', 'left', s)
    s = re.sub(r'\btehre\b', 'there', s)
    s = re.sub(r'\brigth\b', 'right', s)
    s = re.sub(r'\brght\b', 'right', s)
    s = re.sub(r'\bbehine\b', 'behind', s)
    s = re.sub(r'\btv\b', 'TV', s)
    s = re.sub(r'\bchai\b', 'chair', s)
    s = re.sub(r'\bwasing\b', 'washing', s)
    s = re.sub(r'\bwaslked\b', 'walked', s)
    s = re.sub(r"\boclock\b", "o'clock", s)
    s = re.sub(r"\bo'[ ]+clock\b", "o'clock", s)

    s = re.sub(r'\b0\b', 'zero', s)
    s = re.sub(r'\bnone\b', 'zero', s)
    s = re.sub(r'\b1\b', 'one', s)
    s = re.sub(r'\b2\b', 'two', s)
    s = re.sub(r'\b3\b', 'three', s)
    s = re.sub(r'\b4\b', 'four', s)
    s = re.sub(r'\b5\b', 'five', s)
    s = re.sub(r'\b6\b', 'six', s)
    s = re.sub(r'\b7\b', 'seven', s)
    s = re.sub(r'\b8\b', 'eight', s)
    s = re.sub(r'\b9\b', 'nine', s)
    s = re.sub(r'\b10\b', 'ten', s)
    s = re.sub(r'\b11\b', 'eleven', s)
    s = re.sub(r'\b12\b', 'twelve', s)
    s = re.sub(r'\b13\b', 'thirteen', s)
    s = re.sub(r'\b14\b', 'fourteen', s)
    s = re.sub(r'\b15\b', 'fifteen', s)
    s = re.sub(r'\b16\b', 'sixteen', s)
    s = re.sub(r'\b17\b', 'seventeen', s)
    s = re.sub(r'\b18\b', 'eighteen', s)
    s = re.sub(r'\b19\b', 'nineteen', s)
    s = re.sub(r'\b20\b', 'twenty', s)
    s = re.sub(r'\b23\b', 'twenty-three', s)

    s = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', s) 
    s = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', s)      
    s = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', s)     
    s = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', s)    
    s = re.sub(r'\bbackwards\b', 'backward', s)

    return s

def sqa_process_results(doc, results):
    pred =  clean_single_answer(results[0])
    doc['prediction'] = pred
    answers = doc['answer']
    if pred == answers:
        doc['score'] = 1
    elif pred in answers:
        doc['score'] = 1
    elif ''.join(pred.split()) in ''.join(answers.split()):
        doc['score'] = 1
    elif len(set(pred.split()).intersection(answers.split())) > 0:
        doc['score'] = 1
    else:
        doc['score'] = 0
    return {"sqa": doc}

def sqa_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}
    output['accuracy'] = results['score'].mean()
    return output