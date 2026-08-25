# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the Licmmense at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import copy
import json
import random
import math
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

import cv2
import decord
import imageio
import traceback
import numpy as np
import transformers
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

# Add current directory and parent directories to sys.path
current_file_path = os.path.abspath(__file__)
trace_pkg_dir = os.path.dirname(current_file_path) # .../trace
repo_root = os.path.dirname(trace_pkg_dir)         # .../Trace (MSLoc/Trace)

if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from trace import conversation as conversation_lib
from trace.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_INDEX_TOKEN
from trace.trace_trainer import TraceTrainer
from trace.model import *
from trace.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, tokenizer_MMODAL_token_all, process_video_ref_split
import trace.mm_utils as mm_utils_module

from trace.model.multimodal_projector.builder import load_mm_projector

os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_embed_head: bool = field(default=False)
    tune_lm_embed_head: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Other Arguments
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    downsample_num: int = field(default=1)
    
    # Closs Arguments
    closs: bool = field(default=False, metadata={"help": "Whether to use closs (contrastive loss for classification tokens)."})


@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)

    # Train Mode
    # - loc: current behavior (use full video; timestamps are aligned to sampled frame timestamps)
    # - ref: window-based cropping for real/fake videos (see LazySupervisedDataset.__getitem__)
    # - ref2: proposal-based training (load proposals from proposal_path, match with GT)
    train_mode: str = field(default="loc", metadata={"help": "Training mode for dataset sampling: loc, ref, or ref2."})

    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    sample_scheme: Optional[str] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'
    
    # Ref Mode Sampling Arguments
    bnd_ratio: float = field(default=0.2, metadata={"help": "Ratio of boundary region (each side)."})
    bnd_frames: int = field(default=16, metadata={"help": "Number of frames to sample from each boundary region."})
    seg_frames: int = field(default=8, metadata={"help": "Number of frames to sample from the event region."})
    
    # Ref2 Mode Arguments
    proposal_path: str = field(default=None, metadata={"help": "Path to the proposal file for ref2 mode."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'score', 'time', 'sync']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head') 
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        rank0_print('Saving...')
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data


        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # NOTE: scan token of each modal and move them to the beginning of the sentence. 
            for DEFAULT_TOKEN in DEFAULT_MMODAL_TOKEN.values():
                MODAL_TYPE = None
                if DEFAULT_TOKEN in sentence['value'] and 'time' not in DEFAULT_TOKEN and 'score' not in DEFAULT_TOKEN and 'sync' not in DEFAULT_TOKEN:
                    MODAL_TYPE = DEFAULT_TOKEN[1:-1]
                    sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, f'<{MODAL_TYPE.capitalize()}>' + DEFAULT_TOKEN + f'</{MODAL_TYPE.capitalize()}>')
                replace_token = DEFAULT_TOKEN
                if data_args.mm_use_im_start_end and MODAL_TYPE is not None:
                    replace_token = DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()] + replace_token + DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()]
                sentence["value"] = sentence["value"].replace(DEFAULT_TOKEN, replace_token)

    return sources

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 1. Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 2. Tokenize conversations
    if len(MODAL_list) > 0:
        input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN

    # 3. Prepare training inputs and labels.
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        cur_len = 0
        rounds = conversation.split(conv.sep)
        # 3.1 Ignore system prompt (zero order round)
        round_len = len(tokenizer(rounds[0]).input_ids) + 1
        target[cur_len:cur_len+round_len] = IGNORE_INDEX
        cur_len += round_len
        rounds = rounds[1:]

        # QA rounds
        for i, rou in enumerate(rounds):
            if rou == "" or rou == '\n':
                break

            role = conv.roles[i % 2]
            parts = rou.split(role)

            assert len(parts) == 2, f"Invalid conversation: {rou}"
            parts[0] += role

            if len(MODAL_list) > 0:
                round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) + 1
                instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i % 2 == 0:
                # 3.2 Ignore role & instruction
                target[cur_len:cur_len+round_len] = IGNORE_INDEX
            else:
                # 3.3 Ignore role & train response
                target[cur_len:cur_len+instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

    # TODO: Fixing this hardcoding for qwen/ChatML template
    if "qwen" in conv.version:
        for input_id, target in zip(input_ids, targets):
            # <|im_start|>, <|im_end|>
            target[input_id == 151644] = 151644
            target[input_id == 151645] = 151645

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if len(MODAL_list) > 0:
        input_ids = torch.stack([tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                round_len = len(tokenizer_MMODAL_token_all(rou, tokenizer))
                instruction_len = len(tokenizer_MMODAL_token_all(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    assert len(sources) == len(MODAL_list)
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        # source is the conversations in the input data
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    print(conversations)

    # Tokenize conversations
    if len(MODAL_list) > 0:
        input_ids = torch.stack([tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    #for conversation, target in zip(conversations, targets):
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            current_sep = sep
            parts = rou.split(current_sep)
            if len(parts) != 2:
                current_sep = conv.roles[1] + ": "
                parts = rou.split(current_sep)
                if len(parts) != 2:
                    break
            parts[0] += current_sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer)) 
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                # fix the issue of tokenization mismatch
                round_len = len(tokenizer_MMODAL_token_all(rou, tokenizer))
                instruction_len = len(tokenizer_MMODAL_token_all(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list=[]
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    DEFAULT_TOKEN = DEFAULT_MMODAL_TOKEN[MODAL_list[0]]
    for source in sources:
        assert len(source) == 2
        source[0]['value'] = DEFAULT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_MMODAL_token_all(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list: list = []
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, MODAL_list)
    # qwen2 conversation style preprocess
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN:
        return preprocess_qwen(sources, tokenizer, MODAL_list)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts, token_index):
        return [len(tokenizer_MMODAL_token_all(prompt, tokenizer)) for prompt in prompts]

    if len(MODAL_list) > 0:
        input_ids = [tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for idx, (target, source) in enumerate(zip(targets, sources)):
        if len(MODAL_list) > 0:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source], MODAL_list[idx])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 class_to_idx: Dict[str, int] = None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self._external_class_to_idx = class_to_idx  # class-name -> idx loaded from feature file

        if getattr(data_args, 'train_mode', 'loc') == 'ref2':
            rank0_print(f"Loading GT data from {data_path}...")
            gt_data = json.load(open(data_path, "r"))
            # Build GT map: video_path -> item
            gt_map = {}
            for item in gt_data:
                vid = item.get('video_path') or item.get('video') or item.get('image_id')
                if vid:
                    gt_map[vid] = item
            
            if not data_args.proposal_path:
                raise ValueError("proposal_path must be provided for ref2 mode.")

            rank0_print(f"Loading Proposal data from {data_args.proposal_path}...")
            proposal_data = json.load(open(data_args.proposal_path, "r"))
            
            self.list_data_dict = []
            for item in proposal_data:
                vid = item.get('video_path') or item.get('video') or item.get('image_id')
                gt_item = gt_map.get(vid)
                
                if not gt_item:
                    continue
                
                # Get proposals
                proposals = []
                if "model_inference" in item and "segment" in item["model_inference"]:
                    proposals = item["model_inference"]["segment"]
                
                for prop in proposals:
                    if not prop or len(prop) < 2: continue
                    # Create a new sample for each proposal
                    new_sample = {
                        "video": vid,
                        "proposal": prop, # [start, end]
                        "gt_item": gt_item, # Contains annotations
                        "type": "ref2_sample" # Marker
                    }
                    self.list_data_dict.append(new_sample)
            
            rank0_print(f"Loaded {len(self.list_data_dict)} proposals for ref2 training.")

        else:
            list_data_dict = json.load(open(data_path, "r"))
            

            if len(list_data_dict) > 0 and 'annotations' in list_data_dict[0]:
                rank0_print("Converting data format...")
                list_data_dict = self.convert_format(list_data_dict)

                # Downsample real data to maintain 1:3 ratio with fake data
                fake_data = [d for d in list_data_dict if d.get('type') == 'fake']
                real_data = [d for d in list_data_dict if d.get('type') != 'fake']
                
                if len(fake_data) > 0:
                    target_real_count = len(fake_data) // 3
                    if len(real_data) > target_real_count:
                        rank0_print(f"Downsampling real data from {len(real_data)} to {target_real_count} (Fake count: {len(fake_data)})")
                        random.seed(42)
                        real_data = random.sample(real_data, target_real_count)
                        list_data_dict = fake_data + real_data
                        random.shuffle(list_data_dict)
                    else:
                        rank0_print(f"Real data count ({len(real_data)}) is already <= 1/3 of fake data ({len(fake_data)}).")

            rank0_print("Formatting inputs...Skip in lazy mode")
            self.list_data_dict = list_data_dict

        # Collect all unique classes for global contrastive loss
        self.all_unique_classes = set(["Normal", "none"])
        rank0_print("Collecting all unique classes from dataset...")
        for item in self.list_data_dict:
            # Check 'classes' directly if available (from convert_format)
            if 'classes' in item:
                for c_list in item['classes']:
                    for c in c_list:
                        self.all_unique_classes.add(c if c else "none")
            # Check 'gt_item' for ref2 mode
            elif 'gt_item' in item:
                gt_item = item['gt_item']
                if 'annotations' in gt_item:
                    for ann in gt_item['annotations']:
                        # Handle potential dict/list inconsistencies as in __getitem__
                        bnd_cot_st = ann.get('bnd_cot_st', [{}])
                        if isinstance(bnd_cot_st, dict): bnd_cot_st = [bnd_cot_st]
                        
                        obj_cot = ann.get('obj_cot', [{}])
                        if isinstance(obj_cot, dict): obj_cot = [obj_cot]
                        
                        bnd_cot_ed = ann.get('bnd_cot_ed', [{}])
                        if isinstance(bnd_cot_ed, dict): bnd_cot_ed = [bnd_cot_ed]

                        # Extract classes safely.
                        # bnd_cot_st / bnd_cot_ed -> 'bnd_class', obj_cot -> 'bnd_sub_class'.
                        def _safe_get_bnd_cls(container):
                            """Read 'bnd_class' from a bnd_cot_st / bnd_cot_ed entry."""
                            if isinstance(container, dict):
                                return container.get('bnd_class', '')
                            if isinstance(container, list) and len(container) > 0:
                                return container[0].get('bnd_class', '')
                            return ''
                        
                        def _safe_get_obj_cls(container):
                            """Read 'bnd_sub_class' from an obj_cot entry."""
                            if isinstance(container, dict):
                                return container.get('bnd_sub_class', '')
                            if isinstance(container, list) and len(container) > 0:
                                return container[0].get('bnd_sub_class', '')
                            return ''

                        c1 = _safe_get_bnd_cls(bnd_cot_st)  # bnd_cot_st -> bnd_class
                        c2 = _safe_get_obj_cls(obj_cot)      # obj_cot -> bnd_sub_class
                        c3 = _safe_get_bnd_cls(bnd_cot_ed)  # bnd_cot_ed -> bnd_class
                        
                        for c in [c1, c2, c3]:
                            if not c: c = "none"
                            self.all_unique_classes.add(c)
        
        self.all_unique_classes = sorted(list(self.all_unique_classes))
        rank0_print(f"Collected {len(self.all_unique_classes)} unique classes from data: {self.all_unique_classes}")
        
        # Pick which class_to_idx to use.
        # Prefer the externally-provided mapping (loaded from the pre-computed
        # feature file) so that Dataset and Model.ClassFeatureBank are aligned.
        if self._external_class_to_idx is not None:
            self.class_to_idx = self._external_class_to_idx
            external_classes = list(self._external_class_to_idx.keys())
            rank0_print(f"[CLoss] Using external class_to_idx from feature file ({len(external_classes)} classes)")
            rank0_print(f"[CLoss] External classes: {external_classes}")
            
            # Report classes that appear in the data but are not in the feature file.
            missing_classes = [c for c in self.all_unique_classes if c not in self.class_to_idx]
            if missing_classes:
                rank0_print(f"[CLoss WARNING] {len(missing_classes)} classes in data are NOT in feature file (will be ignored):")
                for mc in missing_classes[:20]:  # only print the first 20
                    rank0_print(f"  - '{mc}'")
                if len(missing_classes) > 20:
                    rank0_print(f"  ... and {len(missing_classes) - 20} more")
            
            # Use the class list from the feature file as the canonical order.
            self.all_unique_classes = external_classes
        else:
            # No external mapping; fall back to dynamically collected classes.
            # NOTE: this may not match the feature file's class order.
            self.class_to_idx = {}
            for idx, c in enumerate(self.all_unique_classes):
                self.class_to_idx[c] = idx
            rank0_print(f"[CLoss WARNING] No external class_to_idx provided, using dynamically collected classes")
        
        # Pre-tokenize all classes
        self.all_class_ids = []
        for c in self.all_unique_classes:
            encoded = self.tokenizer(c, return_tensors='pt', padding=False, truncation=True, max_length=64)
            self.all_class_ids.append(encoded.input_ids[0])

    def convert_format(self, list_data_dict):
        new_data = []
        for i, item in enumerate(list_data_dict):
            # If conversations already exist, use as is
            # if 'conversations' in item:
            #     new_data.append(item)
            #     continue
            if item['duration'] < 1:
                continue
            video_path = item.get('video_path', '')
            conversations = []
            times = []
            scores = []
            classes = []
            
            # Handle fake data
            if item.get('type') == 'fake':
                # continue
                conversations.append({
                    "from": "human",
                    "value": "<video>\nPlease examine and localize any inconsistencies or obvious signs of forgery in the video, stating their commencement and completion timestamps and provide a succinct explanation."
                })
                
                gpt_value = ""
                for ann in item.get('annotations', []):
                    segment = ann.get('segment', [])
                    
                    if len(segment) != 2:
                        continue
                    if segment[0] < 0:
                        segment[0] = 0.0
                    if segment[1] < 0:
                        print(f"[WARN] Skip annotation with negative end time: {segment}")
                        continue
                    times.append(segment)
                    scores.append([]) # Empty score list
                    
                    # Extract classes:
                    #   bnd_cot_st / bnd_cot_ed -> 'bnd_class'
                    #   obj_cot                 -> 'bnd_sub_class'
                    def _safe_get_bnd_cls_from_container(container):
                        """Read 'bnd_class' from a bnd_cot_st / bnd_cot_ed container."""
                        if container is None:
                            return ''
                        if isinstance(container, dict):
                            return container.get('bnd_class', '')
                        if isinstance(container, list) and len(container) > 0:
                            return container[0].get('bnd_class', '') if isinstance(container[0], dict) else ''
                        return ''
                    
                    def _safe_get_obj_cls_from_container(container):
                        """Read 'bnd_sub_class' from an obj_cot container."""
                        if container is None:
                            return ''
                        if isinstance(container, dict):
                            return container.get('bnd_sub_class', '')
                        if isinstance(container, list) and len(container) > 0:
                            return container[0].get('bnd_sub_class', '') if isinstance(container[0], dict) else ''
                        return ''
                    
                    c1 = _safe_get_bnd_cls_from_container(ann.get('bnd_cot_st'))  # bnd_cot_st -> bnd_class
                    c2 = _safe_get_obj_cls_from_container(ann.get('obj_cot'))      # obj_cot -> bnd_sub_class
                    c3 = _safe_get_bnd_cls_from_container(ann.get('bnd_cot_ed'))  # bnd_cot_ed -> bnd_class
                    temp_classes = [c1, c2, c3]
                    classes.append(temp_classes)
                    
                    caption = ""
                    try:
                        if 'Round4' not in ann["combine_dir"]:
                            if isinstance(ann['bnd_cot_st'], dict):
                                ann['bnd_cot_st'] = [ann['bnd_cot_st']]
                            if isinstance(ann['bnd_cot_ed'], dict):
                                ann['bnd_cot_ed'] = [ann['bnd_cot_ed']]
                            caption += "Temporal forgery.\n"
                            caption += ann['bnd_cot_st'][0].get('bnd_caption', '') + '\n'
                            caption += ann['obj_cot'][0].get('obj_caption', '') + '\n'
                            caption += ann['bnd_cot_ed'][0].get('bnd_caption', '')
                        elif 'Round4' in ann["combine_dir"]:
                            caption += "Spatio-temporal forgery.\n"
                            caption += ann['obj_cot'][0].get('obj_caption', '')
                        else:
                            raise ValueError(f"Unrecognised combine_dir: {ann.get('combine_dir')}")
                    except Exception as e:
                        print(f"[WARN] Failed to build caption for ann={ann}: {e}")
                        continue

                    time_tokens = "<time>" * 14
                    gpt_value += f"<sync>{time_tokens}<score>{caption}"
                
                conversations.append({
                    "from": "gpt",
                    "value": gpt_value
                })
            
            # Skip non-fake data.
            else:
                continue

            new_item = {
                "video": video_path,
                "id": i,
                "conversations": conversations,
                "scores": scores,
                "times": times,
                "classes": classes
            }
            if 'type' in item:
                new_item['type'] = item['type']
                
            new_data.append(new_item)
        return new_data

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 513 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # cur_len = sum(sum(conv['value'].count(k) for k in MMODAL_INDEX_TOKEN.values()) for conv in sample['conversations'])
            # length_list.append(cur_len)

            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames
        sample_scheme = 'uniform' if self.data_args.sample_scheme is None else self.data_args.sample_scheme
        
        # Handle ref2_sample
        if sources.get('type') == 'ref2_sample':
            video_file = sources['video']
            video_file = os.path.join(self.data_args.data_folder, video_file)
            proposal = sources['proposal'] # [start, end]
            gt_item = sources['gt_item']
            
            win_start, win_end = float(proposal[0]), float(proposal[1])
            
            # Match with GT
            gt_annotations = gt_item.get('annotations', [])
            updated_times = []
            captions = []
            matched_classes = []
            
            for ann in gt_annotations:
                seg = ann.get('segment', [])
                if len(seg) != 2: continue
                s0, s1 = float(seg[0]), float(seg[1])
                
                # Intersection
                inter0 = max(s0, win_start)
                inter1 = min(s1, win_end)
                
                if inter1 > inter0: # Has intersection
                    # Relative time
                    rel_s = inter0 - win_start
                    rel_e = inter1 - win_start
                    updated_times.append([rel_s, rel_e])
                    
                    temp_classes = ['', '', '']
                    # Get caption
                    caption = ""
                    try:
                        if 'Round4' not in ann.get("combine_dir", ""):
                            if isinstance(ann.get('bnd_cot_st'), dict): ann['bnd_cot_st'] = [ann['bnd_cot_st']]
                            if isinstance(ann.get('bnd_cot_ed'), dict): ann['bnd_cot_ed'] = [ann['bnd_cot_ed']]
                            caption += "Temporal forgery.\n"
                            caption += ann.get('bnd_cot_st', [{}])[0].get('bnd_caption', '') + '\n'
                            caption += ann.get('obj_cot', [{}])[0].get('obj_caption', '') + '\n'
                            caption += ann.get('bnd_cot_ed', [{}])[0].get('bnd_caption', '')
                            # bnd_cot_st/bnd_cot_ed -> bnd_class, obj_cot -> bnd_sub_class
                            _c1 = ann.get('bnd_cot_st', [{}])[0].get('bnd_class', '')
                            _c2 = ann.get('obj_cot', [{}])[0].get('bnd_sub_class', '')
                            _c3 = ann.get('bnd_cot_ed', [{}])[0].get('bnd_class', '')
                            temp_classes = [_c1, _c2, _c3]
                            # Warn when all class slots are empty.
                            if not any(temp_classes):
                                print(f"[WARN] Empty classes detected! bnd_cot_st={ann.get('bnd_cot_st')}, obj_cot={ann.get('obj_cot')}, bnd_cot_ed={ann.get('bnd_cot_ed')}", flush=True)
                        elif 'Round4' in ann.get("combine_dir", ""):
                            caption += "Spatio-temporal forgery.\n"
                            caption += ann.get('obj_cot', [{}])[0].get('obj_caption', '')
                            # obj_cot -> bnd_sub_class
                            _c2 = ann.get('obj_cot', [{}])[0].get('bnd_sub_class', '')
                            temp_classes = ['', _c2, '']
                            # Warn when the obj_cot class is empty.
                            if not _c2:
                                print(f"[WARN] Empty obj_cot class detected! obj_cot={ann.get('obj_cot')}", flush=True)
                        matched_classes.append(temp_classes)
                    except Exception as e:
                        print(f"[WARN] Failed to extract classes for ann={ann}: {e}")
                        continue
                    
                    captions.append(caption)

            # Build conversation
            conv = []
            conv.append({
                "from": "human",
                "value": "<video>\nPlease examine and localize any inconsistencies or obvious signs of forgery in the video, stating their commencement and completion timestamps and provide a succinct explanation."
            })
            
            if len(updated_times) > 0:
                # Fake
                gpt_value = ""
                for idx, t_seg in enumerate(updated_times):
                    time_tokens = "<time>" * 14 
                    gpt_value += f"<sync>{time_tokens}<score>{captions[idx]}"
                
                conv.append({
                    "from": "gpt",
                    "value": gpt_value
                })
                times = updated_times
                scores = [[] for _ in updated_times] # Empty scores
            else:
                # Real
                gpt_value = "<sync><time><score>No forgery."
                conv.append({
                    "from": "gpt",
                    "value": gpt_value
                })
                times = [[]]
                scores = [[]]
                final_classes = ['Normal', 'Normal', 'Normal']

            # Process video
            try:
                video, video_timestamps = process_video_ref_split(
                    video_file,
                    video_processor,
                    self.data_args.image_aspect_ratio,
                    bnd_frames=self.data_args.bnd_frames,
                    seg_frames=self.data_args.seg_frames,
                    bnd_ratio=self.data_args.bnd_ratio,
                    start_time=win_start,
                    end_time=win_end
                )
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
                raise e

            # Preprocess conversation
            sources_list = [{"conversations": conv}]
            sources_list = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources_list]), self.data_args)
            MODAL_list = ['VIDEO']
            
            data_dict = preprocess(sources_list, self.tokenizer, MODAL_list=MODAL_list)
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
            
            data_dict['time'] = times
            data_dict['score'] = scores
            data_dict['video'] = video
            data_dict['video_timestamps'] = video_timestamps
            
            # Handle classes for closs.
            # If matched_classes is empty, the proposal window does not overlap
            # any forged segment, so the sample should be treated as "Real" and
            # mapped to the Normal class.
            if len(matched_classes) > 0:
                # Fake sample: pick the first non-empty class set.
                final_classes = ['', '', '']
                for cls_set in matched_classes:
                    if any(c for c in cls_set):
                        final_classes = cls_set
                        break

                # Ensure no empty classes, default to "none"
                final_classes = [c if c else "none" for c in final_classes]
            else:
                # Real sample: no overlap with any forged segment.
                final_classes = ['Normal', 'Normal', 'Normal']
            data_dict['classes'] = final_classes
            
            # Map classes to indices
            class_indices = []
            for c in final_classes:
                if c in self.class_to_idx:
                    class_indices.append(self.class_to_idx[c])
                else:
                    class_indices.append(-100) # Ignore index
            
            # Verify that at least one valid class index (not -100) is present.
            valid_class_count = sum(1 for idx in class_indices if idx != -100)
            if valid_class_count == 0:
                # Collect raw data for the error message.
                import json as _json
                _debug_sources_str = _json.dumps(sources, indent=2, ensure_ascii=False, default=str)
                _debug_gt_annotations = sources.get('gt_item', {}).get('annotations', [])
                _debug_annotations_str = _json.dumps(_debug_gt_annotations, indent=2, ensure_ascii=False, default=str)
                
                error_msg = f"""
================================================================================
Sample has no valid class for CLoss training.
================================================================================

[Basic info]
  - Sample index: {i}
  - final_classes: {final_classes}
  - class_indices: {class_indices}
  - matched_classes (all): {matched_classes}
  - class_to_idx keys (first 20): {list(self.class_to_idx.keys())[:20]}...

[Video info]
  - video_file: {sources.get('video', 'N/A')}
  - proposal: {sources.get('proposal', 'N/A')}

[Raw gt_item.annotations]
{_debug_annotations_str}

[Full sources]
{_debug_sources_str}

[Hints]
  1. Check that bnd_cot_st / bnd_cot_ed entries have a 'bnd_class' field.
  2. Check that obj_cot entries have a 'bnd_sub_class' field.
  3. Check that the class name exists in class_to_idx.
================================================================================
"""
                if not getattr(self, '_warned_no_valid_closs_class', False):
                    print(f"[CLoss WARNING] {error_msg}", flush=True)
                    self._warned_no_valid_closs_class = True
            
            data_dict['class_indices'] = torch.tensor(class_indices, dtype=torch.long)

            # Map times to timestamps
            if len(times) > 0 and len(times[0]) > 0:
                 data_dict['time'] = [[min(data_dict['video_timestamps'], key=lambda x: abs(x[0] - target))[0] for target in interval] for interval in data_dict['time']]
            
            return data_dict

        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames
        sample_scheme = 'uniform' if self.data_args.sample_scheme is None else self.data_args.sample_scheme

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        ##############################################################################################################

        times = []
        scores = []
        classes = []
        times_count = 0
        scores_count = 0
        for e in sources:
            times.extend(e.get('times', []))
            scores.extend(e.get('scores', []))
            classes.extend(e.get('classes', []))

        def _normalize_class_triplet(class_values, fallback):
            if isinstance(class_values, (list, tuple)):
                normalized = list(class_values[:3])
            else:
                normalized = [class_values]
            normalized = (normalized + ['', '', ''])[:3]
            if any(c for c in normalized):
                return [c if c else "none" for c in normalized]
            return fallback

        sample_type = self.list_data_dict[i].get('type', None) if isinstance(i, int) else sources[0].get('type', None)
        if sample_type == 'real' or len(times) == 0:
            final_classes = ['Normal', 'Normal', 'Normal']
        else:
            final_classes = ['none', 'none', 'none']
            for cls_set in classes:
                candidate_classes = _normalize_class_triplet(cls_set, ['none', 'none', 'none'])
                if any(c != "none" for c in candidate_classes):
                    final_classes = candidate_classes
                    break

        # Check times for validity
        for t_seg in times:
            if len(t_seg) >= 2:
                try:
                    s0, s1 = float(t_seg[0]), float(t_seg[1])
                    if math.isnan(s0) or math.isnan(s1) or math.isinf(s0) or math.isinf(s1):
                        raise ValueError(f"Invalid time segment found: {t_seg}")
                except (ValueError, TypeError) as e:
                    print(f"Error parsing time segment {t_seg}: {e}")
                    raise e

        ##############################################################################################################
        MODAL_list = []
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_file = os.path.join(self.data_args.data_folder, image_file)

            try:
                image = process_image(image_file, image_processor, self.data_args.image_aspect_ratio)[0]
            except Exception as e:
                print(f"Encounted error when reading image {image_file}")
                raise e

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            MODAL_list.append('IMAGE')
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            # video_file = '__5RJw4UP1Y.mp4'
            video_file = os.path.join(self.data_args.data_folder, video_file)

            def _clamp(v, lo, hi):
                return max(lo, min(hi, v))

            def _sample_ref_window_for_real(duration_s: float):
                # random window in [3, 15] seconds
                win_len = random.uniform(3.0, 15.0)
                win_len = min(win_len, max(0.1, duration_s))
                if duration_s <= win_len:
                    return 0.0, duration_s
                start = random.uniform(0.0, duration_s - win_len)
                return start, start + win_len

            def _sample_ref_window_for_fake(duration_s: float, seg: list):
                # seg: [seg_start, seg_end] in seconds
                seg_start, seg_end = float(seg[0]), float(seg[1])
                if seg_end < seg_start:
                    seg_start, seg_end = seg_end, seg_start
                seg_start = _clamp(seg_start, 0.0, duration_s)
                seg_end = _clamp(seg_end, 0.0, duration_s)
                if seg_end <= seg_start:
                    # degenerate, fallback to real-like
                    return _sample_ref_window_for_real(duration_s), [0.0, 0.0]

                x = seg_end - seg_start
                # window length in [x/2, 2x] AND [3, 20]
                lo = max(3.0, x / 2.0)
                hi = min(20.0, 2.0 * x)
                if hi < lo:
                    # if x is too small/too large, degrade to [3,20] but still must intersect
                    lo = max(0.1, min(3.0, duration_s))
                    hi = max(lo, min(20.0, duration_s))

                win_len = random.uniform(lo, hi)
                win_len = min(win_len, max(0.1, duration_s))
                if duration_s <= win_len:
                    win_start, win_end = 0.0, duration_s
                else:
                    # constraints:
                    # 1) window intersects [seg_start, seg_end]
                    # 2) window within [0, duration_s]
                    # 3) |win_start - seg_start| <= win_len AND |win_end - seg_end| <= win_len
                    #
                    # Intersection implies: win_start <= seg_end and win_end >= seg_start
                    # with win_end = win_start + win_len.
                    # So win_start in [seg_start - win_len, seg_end]
                    # plus in [0, duration_s - win_len]
                    # and constraints 3) translate to:
                    #   win_start >= seg_start - win_len
                    #   win_start <= seg_start + win_len
                    #   win_start >= seg_end - 2*win_len
                    #   win_start <= seg_end
                    lb = max(0.0, seg_start - win_len, seg_end - 2.0 * win_len)
                    ub = min(duration_s - win_len, seg_end, seg_start + win_len)
                    if ub < lb:
                        # fallback: enforce intersection only
                        lb = max(0.0, seg_start - win_len)
                        ub = min(duration_s - win_len, seg_end)
                    if ub < lb:
                        win_start = _clamp(seg_start, 0.0, max(0.0, duration_s - win_len))
                    else:
                        win_start = random.uniform(lb, ub)
                    win_end = win_start + win_len
                    win_end = _clamp(win_end, 0.0, duration_s)
                    win_start = _clamp(win_start, 0.0, win_end)

                # compute new segment inside window
                new_seg_start = _clamp(seg_start - win_start, 0.0, win_end - win_start)
                new_seg_end = _clamp(seg_end - win_start, 0.0, win_end - win_start)
                # keep only intersected part
                new_seg_start = _clamp(new_seg_start, 0.0, win_end - win_start)
                new_seg_end = _clamp(new_seg_end, 0.0, win_end - win_start)
                if new_seg_end < new_seg_start:
                    new_seg_start, new_seg_end = new_seg_end, new_seg_start
                return (win_start, win_end), [new_seg_start, new_seg_end]

            try:
                if getattr(self.data_args, 'train_mode', 'loc') == 'ref':
                    # Determine video duration from metadata (fast, without decoding frames)
                    try:
                        if not os.path.exists(video_file):
                            raise FileNotFoundError(f"Video file not found: {video_file}")
                        if os.path.getsize(video_file) == 0:
                            raise ValueError(f"Video file is empty: {video_file}")

                        from decord import VideoReader, cpu
                        vr_meta = VideoReader(uri=video_file, ctx=cpu(0))
                        duration_frames = len(vr_meta)
                        fps = float(vr_meta.get_avg_fps())
                        if fps <= 0 or math.isnan(fps) or math.isinf(fps):
                            fps = 30.0
                        duration_s = duration_frames / fps
                    except Exception:
                        # If metadata fails, fallback to full decode duration via process_video later
                        duration_s = None

                    cur_type = self.list_data_dict[i].get('type', None)
                    # choose window
                    win_start, win_end = 0.0, None
                    updated_times = None
                    if duration_s is not None and duration_s > 0:
                        if cur_type == 'real':
                            win_start, win_end = _sample_ref_window_for_real(duration_s)
                            final_classes = ['Normal', 'Normal', 'Normal']
                        elif cur_type == 'fake' and len(times) > 0:
                            chosen_idx = random.choice(range(len(times)))
                            chosen_seg = times[chosen_idx]
                            if chosen_idx < len(classes):
                                final_classes = _normalize_class_triplet(classes[chosen_idx], ['none', 'none', 'none'])
                            (win_start, win_end), new_seg = _sample_ref_window_for_fake(duration_s, chosen_seg)
                            # Replace all segments with their intersected parts within the chosen window.
                            # Segments not intersecting will become [0,0] and will be ignored by downstream if desired.
                            updated_times = []
                            valid_indices = []
                            for idx, seg in enumerate(times):
                                s0, s1 = float(seg[0]), float(seg[1])
                                inter0 = max(s0, win_start)
                                inter1 = min(s1, win_end)
                                if inter1 <= inter0:
                                    # no intersection
                                    continue
                                updated_times.append([inter0 - win_start, inter1 - win_start])
                                valid_indices.append(idx)
                            # Ensure the chosen fake segment contributes at least one segment
                            if len(updated_times) == 0:
                                updated_times = [new_seg]
                                valid_indices = [chosen_idx]
                        else:
                            # unknown type or no annotations -> fallback to real window
                            win_start, win_end = _sample_ref_window_for_real(duration_s)
                            final_classes = ['Normal', 'Normal', 'Normal']

                        # decode/crop video frames and timestamps (STRICTLY within window)
                        # process_video will sample EXACTLY `num_frames` frames inside [win_start, win_end]
                        # and return timestamps in the window coordinate system (start at 0).
                        video, video_timestamps = process_video_ref_split(
                            video_file,
                            video_processor,
                            self.data_args.image_aspect_ratio,
                            bnd_frames=self.data_args.bnd_frames,
                            seg_frames=self.data_args.seg_frames,
                            bnd_ratio=self.data_args.bnd_ratio,
                            start_time=win_start,
                            end_time=win_end,
                            vr=vr_meta,
                        )

                        # update times and scores for fake case
                        if updated_times is not None:
                            times = updated_times
                            # Sync `scores` so it keeps only the entries pointed to by valid_indices.
                            if valid_indices and len(scores) > 0:
                                scores = [scores[idx] for idx in valid_indices if idx < len(scores)]
                            if cur_type == 'fake':
                                conv = sources[0].get('conversations', [])
                                for msg in conv:
                                    if msg['from'] == 'gpt':
                                        parts = msg['value'].split('<sync>')
                                        if len(parts) > 1:
                                            new_value = ""
                                            for idx in valid_indices:
                                                if idx + 1 < len(parts):
                                                    new_value += "<sync>" + parts[idx+1]
                                            msg['value'] = new_value
                    else:
                        # duration unknown, fallback to loc
                        video, video_timestamps = process_video(video_file, video_processor, self.data_args.image_aspect_ratio, num_frames, sample_scheme=sample_scheme)
                else:
                    video, video_timestamps = process_video(video_file, video_processor, self.data_args.image_aspect_ratio, num_frames, sample_scheme=sample_scheme)
            except Exception as e:
                print(f"Encounted error when reading video {video_file}")
                raise e

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            MODAL_list.append('VIDEO')
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # NOTE: for sharegpt data in the sft stage, we use the default IMAGE as modal token
            MODAL_list.append('IMAGE')

        data_dict = preprocess(sources, self.tokenizer, MODAL_list=MODAL_list)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        ##############################################################################################################

        data_dict['time'] = times
        data_dict['score'] = scores
        data_dict['classes'] = final_classes

        # Map classes to indices
        class_indices = []
        for c in final_classes:
            if hasattr(self, 'class_to_idx') and c in self.class_to_idx:
                class_indices.append(self.class_to_idx[c])
            else:
                class_indices.append(-100)
        
        # Verify that at least one valid class index (not -100) is present.
        valid_class_count = sum(1 for idx in class_indices if idx != -100)
        if valid_class_count == 0:
            if not getattr(self, '_warned_no_valid_closs_class', False):
                print(
                    "Sample has no valid class for CLoss training. "
                    "The sample will use ignore labels for CLoss.\n"
                    f"  - final_classes: {final_classes}\n"
                    f"  - class_indices: {class_indices}\n"
                    f"  - Sample index: {i}\n"
                    f"  - Available classes (first 10): {list(self.class_to_idx.keys())[:10]}...",
                    flush=True,
                )
                self._warned_no_valid_closs_class = True
        
        data_dict['class_indices'] = torch.tensor(class_indices, dtype=torch.long)

        ##############################################################################################################

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['video_timestamps'] =[[0]] * num_frames
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
            data_dict['video_timestamps'] = video_timestamps
            data_dict['time'] = [[min(data_dict['video_timestamps'], key=lambda x: abs(x[0] - target))[0] for target in interval] for interval in data_dict['time']]
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    all_class_ids: Optional[List[torch.Tensor]] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        Xs, keys = [], []
        ##############################################################################################################
        times, scores, video_timestamps = [], [], []
        closs_input_ids = []
        for instance in instances:
            for x in DEFAULT_MMODAL_TOKEN.keys():
                x = x.lower()
                if x in instance:
                    if 'time' in x:
                        times.append(instance[x])
                    elif 'score' in x:
                        scores.append(instance[x])
                    elif 'sync' in x:
                        continue
                    else:
                        Xs.append(instance[x])
                        keys.append(x)
            video_timestamps.append(instance['video_timestamps'])
            
            # Handle classes
            cur_classes = instance.get('classes', ['', '', ''])
            cur_ids = []
            for cls_str in cur_classes:
                if cls_str:
                    # Use tokenizer
                    encoded = self.tokenizer(cls_str, return_tensors='pt', padding=False, truncation=True, max_length=64)
                    cur_ids.append(encoded.input_ids[0])
                else:
                    cur_ids.append(torch.tensor([], dtype=torch.long))
            closs_input_ids.append(cur_ids)

        # Handle closs_labels (indices into all_class_input_ids)
        closs_labels = []
        for instance in instances:
            if 'class_indices' in instance:
                closs_labels.append(instance['class_indices'])
            else:
                closs_labels.append(torch.full((3,), -100, dtype=torch.long))
        batch['closs_labels'] = torch.stack(closs_labels)

        # Pad closs_input_ids
        max_len = 0
        for item in closs_input_ids:
            for ids in item:
                max_len = max(max_len, len(ids))
        
        if max_len > 0:
            padded_batch = []
            for item in closs_input_ids:
                padded_item = []
                for ids in item:
                    pad_len = max_len - len(ids)
                    if pad_len > 0:
                        padded = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                    else:
                        padded = ids
                    padded_item.append(padded)
                padded_batch.append(torch.stack(padded_item))
            batch['closs_input_ids'] = torch.stack(padded_batch) # B x 3 x L
        else:
            batch['closs_input_ids'] = None

        batch['images'] = [Xs, keys]  # we do not change the key's name.
        batch['times'] = times
        batch['scores'] = scores
        batch['video_timestamps'] = video_timestamps

        # Process all_class_ids if available
        if self.all_class_ids is not None:
            # Pad all_class_ids
            max_len = 0
            for ids in self.all_class_ids:
                max_len = max(max_len, len(ids))
            
            padded_all_classes = []
            for ids in self.all_class_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    padded = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                else:
                    padded = ids
                padded_all_classes.append(padded)
            batch['all_class_input_ids'] = torch.stack(padded_all_classes) # K x L
        else:
            batch['all_class_input_ids'] = None

        return batch
        ##############################################################################################################


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                class_to_idx: Dict[str, int] = None) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    
    Args:
        tokenizer: Tokenizer for text encoding
        data_args: Data arguments
        class_to_idx: Optional mapping from class name to index (loaded from
                      the pre-computed feature file). If provided, the
                      Dataset uses this mapping instead of building one
                      dynamically.
    """
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        class_to_idx=class_to_idx
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, all_class_ids=getattr(train_dataset, 'all_class_ids', None))
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation="eager"):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    def apply_runtime_model_config(config):
        config._attn_implementation = attn_implementation
        config.downsample_num = model_args.downsample_num
        config.closs = model_args.closs
        if model_args.vision_tower is not None:
            config.mm_vision_tower = model_args.vision_tower
            config.vision_tower = model_args.vision_tower
        return config

    if model_args.vision_tower is not None:
        if 'vicuna' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config = apply_runtime_model_config(config)
            model = TraceLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'mixtral' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config = apply_runtime_model_config(config)
            model = TraceMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
            if training_args.deepspeed:
                import deepspeed
                deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif 'qwen2' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config = apply_runtime_model_config(config)
            model = TraceQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        else:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config = apply_runtime_model_config(config)
            model = TraceMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                # ignore_mismatched_sizes=True,
                **bnb_model_from_pretrained_args
            )
    else:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config = apply_runtime_model_config(config)
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        print('Freezed Backbone')
        model.model.requires_grad_(False)
    else:
        model.model.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None: 
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            if model_args.version == "v1":
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            elif model_args.version == "v1_mistral":
                conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]

    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        ##################################################################################################################
        model.config.tune_mm_embed_head = training_args.tune_mm_embed_head = model_args.tune_mm_embed_head
        model.config.tune_lm_embed_head = training_args.tune_lm_embed_head = model_args.tune_lm_embed_head
        ##################################################################################################################
        if model_args.tune_mm_mlp_adapter:
            if model_args.freeze_backbone:
                model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        if model_args.mm_projector_type == 'ref_projector':
            try:
                mm_projector_weights = load_mm_projector(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
                
                # Detect prefix
                keys = list(mm_projector_weights.keys())
                prefix = ''
                if any('model.mm_projector.' in k for k in keys):
                    prefix = 'model.mm_projector.'
                elif any('mm_projector.' in k for k in keys):
                    prefix = 'mm_projector.'

                # Skip if weights are already from a RefProjector.
                if any('boundary_proj' in k for k in keys):
                    pass
                else:
                    def copy_weights(src_prefix, dst_module):
                        with torch.no_grad():
                            # slots
                            if hasattr(dst_module, 'slots'):
                                src_slots = mm_projector_weights.get(f'{src_prefix}slots')
                                if src_slots is not None:
                                    if dst_module.slots.shape == src_slots.shape:
                                        dst_module.slots.copy_(src_slots)
                                    else:
                                        # Repeat to match shape
                                        repeat_factor = (dst_module.slots.shape[1] + src_slots.shape[1] - 1) // src_slots.shape[1]
                                        src_slots_repeated = src_slots.repeat(1, repeat_factor)
                                        dst_module.slots.copy_(src_slots_repeated[:, :dst_module.slots.shape[1]])
                            
                            # ln_vision
                            if hasattr(dst_module, 'ln_vision'):
                                w = mm_projector_weights.get(f'{src_prefix}ln_vision.weight')
                                b = mm_projector_weights.get(f'{src_prefix}ln_vision.bias')
                                if w is not None: dst_module.ln_vision.weight.copy_(w)
                                if b is not None: dst_module.ln_vision.bias.copy_(b)
                            
                            # readout
                            if hasattr(dst_module, 'readout'):
                                w = mm_projector_weights.get(f'{src_prefix}readout.weight')
                                if w is not None: dst_module.readout.weight.copy_(w)

                    # Copy to boundary_proj and event_proj.
                    copy_weights(prefix, model.get_model().mm_projector.boundary_proj)
                    copy_weights(prefix, model.get_model().mm_projector.event_proj)

            except Exception as e:
                rank0_print(f"Failed to initialize RefProjector: {e}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    # ##############################################################################################################

    # #initialize time and score towers
    rank0_print('Initializing time and score towers')
    model.get_model().initialize_time_modules(model_args=model_args)
    model.get_model().initialize_score_modules(model_args=model_args)

    # ##############################################################################################################
    # Initialize CLoss modules (only when train_mode == 'ref2' and closs=True).
    # Note: `closs` is defined under ModelArguments while `train_mode` lives in DataArguments.
    if getattr(model_args, 'closs', False) and getattr(data_args, 'train_mode', 'loc') == 'ref2':
        rank0_print('Initializing CLoss modules from pre-computed features...')
        
        # Load class features from a pre-computed bge-large-en-v1.5 feature file
        # (run extract_class_features.py first to generate it).
        model.get_model().initialize_closs_modules(
            class_feature_path=getattr(data_args, 'class_feature_path', None),  # optional override
            hidden_size=model.config.hidden_size
        )
        
        # Class list loaded from the feature file.
        model_class_names = getattr(model.get_model(), 'class_names', [])
        model_class_to_idx = getattr(model.get_model(), 'class_to_idx', {})
        rank0_print(f'[CLoss] Loaded {len(model_class_names)} classes from feature file: {model_class_names}')
        
        # Persist into config so inference can reuse the same mapping.
        model.config.closs_class_names = model_class_names
        model.config.closs_class_to_idx = model_class_to_idx
    # ##############################################################################################################

    for n, p in model.named_parameters():
        if getattr(training_args, 'tune_mm_embed_head', False):
            if 'time' in n or 'score' in n or 'sync' in n:
                p.requires_grad = True
        else:
            if 'time' in n or 'score' in n or 'sync' in n:
                p.requires_grad = False
        if getattr(training_args, 'tune_lm_embed_head', False):
            if ('head' in n or 'embed_tokens' in n) and ('time' not in n) and ('score' not in n) and ('sync' not in n):
                p.requires_grad = True
        else:
            if ('head' in n or 'embed_tokens' in n) and ('time' not in n) and ('score' not in n) and ('sync' not in n):
                p.requires_grad = False

        if 'closs_tokens' in n:
            p.requires_grad = True
        
        # CLoss module parameters: ClossProjector is trainable, ClassFeatureBank is frozen.
        if 'closs_projector' in n or 'class_feature_bank' in n:
            if 'class_feature_bank' in n:
                p.requires_grad = False  # frozen
            else:
                p.requires_grad = True  # trainable

        if 'lora' in n:
            p.requires_grad = True
        if p.requires_grad == True:
            rank0_print(n)

    # Pull class_to_idx from the model's ClassFeatureBank (when CLoss is enabled).
    # This keeps the Dataset's class indices aligned with the model.
    model_class_to_idx = None
    if getattr(model_args, 'closs', False):
        model_class_to_idx = getattr(model.get_model(), 'class_to_idx', None)
        if model_class_to_idx is not None:
            rank0_print(f"[CLoss] Passing model's class_to_idx to Dataset ({len(model_class_to_idx)} classes from feature file)")
        else:
            rank0_print("[CLoss WARNING] Model does not have class_to_idx, Dataset will use dynamically collected classes")
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, class_to_idx=model_class_to_idx)
    # select a Trainer
    trainer = TraceTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # Force load from model_name_or_path, ignoring existing checkpoints in output_dir
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print(f"WARNING: Found existing checkpoints in {training_args.output_dir}, but forcing training from {model_args.model_name_or_path} as requested.")
    
    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        torch.cuda.synchronize()
        trainer.model = trainer.model.merge_and_unload()
        print(trainer.model)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
