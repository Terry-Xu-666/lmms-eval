import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import re


import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm


from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

import sys ; sys.path.append("LLaVA-Mini")

try:
    from llavamini.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
    from llavamini.conversation import conv_templates, SeparatorStyle
    from llavamini.model.builder import load_pretrained_model
    from llavamini.utils import disable_torch_init
    from llavamini.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
)


except ImportError:
    eval_logger.warning("Failed to import llavamini; Please install it via LLaVA-Mini repository")


@register_model("llava_mini")
class LLaVAMini(lmms):
    """
    LLaVA-Mini
    "https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b"

    For better performance, please visit the LLaVA-Mini repo to get the latest system prompt based on your running tasks.
    https://github.com/ictnlp/LLaVA-Mini
    """

    def __init__(
        self,
        pretrained: str = "ICTNLP/llava-mini-llama-3.1-8b",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "eager",
        max_num_frames: Optional[int] = None,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        conv_template: Optional[str] = "llava_llama_3_1",
        load_8bit: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        

        assert int(batch_size) == 1, "LLaVA-Mini only supports batch size 1"

        self.fps = fps
        if max_num_frames:
            self.max_num_frames = max_num_frames
            self.fps = None
        else:
            self.max_num_frames = None

        self.conv_template = conv_templates[conv_template].copy()

     
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        disable_torch_init()
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(pretrained, None, "llava_mini", load_8bit=False)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._max_length = kwargs.get("max_length", 2048)
        
       

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

    @property
    def image_processor(self):
        return self._image_processor

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_seq_frames(self, total_num_frames, desired_num_frames):
        """
        Calculate the indices of frames to extract from a video.

        Parameters:
        total_num_frames (int): Total number of frames in the video.
        desired_num_frames (int): Desired number of frames to extract.

        Returns:
        list: List of indices of frames to extract.
        """

        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_num_frames - 1) / desired_num_frames

        seq = []
        for i in range(desired_num_frames):
            # Calculate the start and end indices of each segment
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))

            # Append the middle index of the segment to the list
            seq.append((start + end) // 2)

        return seq

    def load_video(self, vis_path):
        """
        Load video frames from a video file.

        Parameters:
        vis_path (str): Path to the video file.
        n_clips (int): Number of clips to extract from the video. Defaults to 1.
        num_frm (int): Number of frames to extract from each clip. Defaults to 100.

        Returns:
        list: List of PIL.Image.Image objects representing video frames.
        """

        # Load video with VideoReader
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        if self.fps:
            video_fps=vr.get_avg_fps()
            num_frm=int(total_frame_num//video_fps*self.fps)
    
            num_frm=min(num_frm,10000)
        else:
            num_frm=self.max_num_frames
    

        # Calculate total number of frames to extract
        total_num_frm = min(total_frame_num, num_frm)
        # Get indices of frames to extract
        frame_idx = self.get_seq_frames(total_frame_num, total_num_frm)
        # Extract frames as numpy array
        img_array = vr.get_batch(frame_idx).asnumpy()
        # Set target image height and width
        target_h, target_w = 336, 336   
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        # Reshape array to match number of clips and frames
        img_array = img_array.reshape(
            (1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

        return clip_imgs

    def split_image(self, image, n=2):
        if n==1: return [image]
        width, height = image.size
        block_width = width // n
        block_height = height // n

        blocks = []

        for i in range(n):
            for j in range(n):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height
                block = image.crop((left, upper, right, lower))
                blocks.append(block)
        blocks.append(image)

        return blocks


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            prompt_list = []
            visual_list = []
            for i,context in enumerate(contexts):
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in context:
                    if self.model.config.mm_use_im_start_end:
                        context = re.sub(IMAGE_PLACEHOLDER, image_token_se, context)
                    else:
                        context = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, context)
                else:
                    if self.model.config.mm_use_im_start_end:
                        context = image_token_se + "\n" + context
                    else:
                        context = DEFAULT_IMAGE_TOKEN + "\n" + context
                
                conv = self.conv_template.copy()
                conv.append_message(conv.roles[0], context)
                conv.append_message(conv.roles[1], None)
                prompt_list.append(conv.get_prompt())
                visual = visuals[i] if i < len(visuals) else None
                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                    video_frames = self.load_video(visual)
                    temporal_len = len(video_frames)
                    
                    N = getattr(self.model.config, 'resolution_ratio', 1)
                    images = []
                    for video_frame in video_frames:
                        images.extend(self.split_image(video_frame, n=N))

                    image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                    image_tensor = image_tensor.unsqueeze(0)

                    bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
                    images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)
                    visual_list.append(images_tensor)
                    
                elif isinstance(visuals[i], Image.Image):
                    image = visual.convert("RGB")
                    N=getattr(self.model.config,'resolution_ratio', 1)
                    images=self.split_image(image,n=N)
                    images_tensor = process_images(
                        images,
                        self.image_processor,
                        self.model.config
                    )
                    visual_list.append(images_tensor.unsqueeze(0))
                else:
                    raise ValueError(f"Unsupported visual type: {type(visual)}")
            

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 512
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = None
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if gen_kwargs["temperature"] > 0:
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
                gen_kwargs["temperature"] = None

            gen_kwargs.pop("until")
            
            answers = []
            for prompt, visual in zip(prompt_list, visual_list):
                try:
                    input_ids = (
                        tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )
                    visual = visual.to(self.model.device, dtype=torch.float16)
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids,
                            images=visual,
                            use_cache=True,
                            **gen_kwargs,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                except Exception as e:
                    eval_logger.error(f"Error {e} in generating")
                    answer = ""
                    res.append(answer)
                    pbar.update(1)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                    continue
               
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                answers.append(outputs)
            
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
