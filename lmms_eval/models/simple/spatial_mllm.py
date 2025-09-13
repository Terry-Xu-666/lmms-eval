import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
import sys; sys.path.append("./spatial-mllm/")
try:
    from src.models import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
except ImportError:
    raise ImportError("spatial_mllm is not installed. Please install it via `pip install spatial_mllm`")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("spatial_mllm")
class Spatial_MLLM(lmms):
    """
    Spatial_MLLM Model
    "https://huggingface.co/Diankun/Spatial-MLLM-subset-sft"
    """

    def __init__(
        self,
        pretrained: str = "Diankun/Spatial-MLLM-subset-sft",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 32,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

      

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_num_frames = max_num_frames

        
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            assert len(visuals) == 1, "VILA only supports one visual input"
            
            media = visuals[0]
            text = contexts
            assert any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]), "Spatial_MLLM only supports video input"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": media,
                            "nframes": self.max_num_frames,
                        },
                        {
                            "type": "text",
                            "text": text,
                        },
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            _, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
            inputs = inputs.to(self.model.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "use_cache" not in gen_kwargs:
                gen_kwargs["use_cache"] = self.use_cache
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            with torch.inference_mode():
                generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        do_sample=gen_kwargs["do_sample"],
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        use_cache=gen_kwargs["use_cache"],
                    )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = self.clean_text(output_text[0])
            print("Answer: ", output_text)
            res.append(output_text)
            pbar.update(1)
        return res
    @staticmethod
    def clean_text(text, exclue_chars=["\n", "\r"]):
        # Extract content between <answer> and </answer> if present
        answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_matches:
            # Use the last match
            text = answer_matches[-1]

        for char in exclue_chars:
            if char in ["\n", "\r"]:
                # If there is a space before the newline, remove the newline
                text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
                # If there is no space before the newline, replace it with a space
                text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
            else:
                text = text.replace(char, " ")

        # Remove leading and trailing spaces and convert to lowercase
        return text.strip().rstrip(".").lower()

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
