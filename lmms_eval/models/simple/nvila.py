import argparse
import json
import logging
import math
import os
import signal
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm
import cv2
from pydantic import BaseModel
from termcolor import colored
from transformers import GenerationConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
try:
    import llava
    from llava.media import Image, Video
   
except ImportError as e:
    raise ImportError(f"VILA is not installed. Please install VILA to use this model. Error: {e}")


@register_model("nvila")
class NVILA(lmms):
    """
    NVILA Model
    """

    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/NVILA-8B",
        max_frames_num: Optional[int] = 32,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model = llava.load(pretrained, model_base=None)
        if max_frames_num > 0:
            self._model.config.num_video_frames = max_frames_num
        # self._config = AutoConfig.from_pretrained(self.pretrained)


        self._config = self._model.config


        self._model.eval()
        self.batch_size_per_gpu = int(batch_size)
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length


    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

 

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            assert len(visuals) == 1, "VILA only supports one visual input"
            
            media = visuals[0]
            text = contexts

            # Prepare multi-modal prompt
            has_video = False
            prompt = []
            if media is not None:
                    if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        media = Image(media)
                    elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                        cap = cv2.VideoCapture(media)
                        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                        media = Video(media)
                        has_video = True
                    else:
                        raise ValueError(f"Unsupported media type: {media}")
                    prompt.append(media)
            if text is not None:
                prompt.append(text)
            generation_config = GenerationConfig(
                max_new_tokens=1024,
                temperature=0.2,
                top_p=None,
                num_beams=1,
            )
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generation_config = GenerationConfig(
                max_new_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
            )

            with torch.inference_mode():
                response = self.model.generate_content(prompt, generation_config=generation_config)

                # output_ids_2 = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=50,stopping_criteria=[stopping_criteria])
                # output_ids = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=50,use_cache=True)

            
            print("Question: ", text)
            print("Answer: ", response)
            res.append(response)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
