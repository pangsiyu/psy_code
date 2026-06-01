# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
from contextlib import nullcontext

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def save_stop_progress_head(model, output_dir: str):
    unwrapped_model = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped_model, "stop_progress_head"):
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    state_dict = {}
    for name, param in unwrapped_model.stop_progress_head.named_parameters():
        gather_context = nullcontext()
        if hasattr(param, "ds_id"):
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters([param], modifier_rank=0)
        with gather_context:
            if rank == 0:
                state_dict[name] = param.detach().cpu().clone()

    if rank == 0:
        for name, buffer in unwrapped_model.stop_progress_head.named_buffers():
            state_dict[name] = buffer.detach().cpu().clone()
        save_path = os.path.join(output_dir, "stop_progress_head.pt")
        torch.save(
            {
                "state_dict": state_dict,
                "stop_head_hidden_dim": getattr(unwrapped_model.config, "stop_head_hidden_dim", None),
                "stop_loss_weight": getattr(unwrapped_model.config, "stop_loss_weight", None),
            },
            save_path,
        )
        print(f"Saved stop_progress_head to {save_path}")


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # vggt is frozen
    for n, p in model.vggt.named_parameters():
        p.requires_grad = False
    for n, p in model.merger.named_parameters():
        p.requires_grad = True
    if hasattr(model, "stop_progress_head"):
        for n, p in model.stop_progress_head.named_parameters():
            p.requires_grad = True
        

VALID_TRAINABLE_SCOPES = {
    "default",
    "stop_head_only",
    "stop_head_merger",
    "stop_head_lora",
    "stop_head_merger_lora",
}


def _is_merger_param(name: str) -> bool:
    return name.startswith("merger.") or ".merger." in name


def _is_lora_param(name: str) -> bool:
    return "lora" in name.lower()


def print_trainable_parameter_summary(model, scope: str) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    rank0_print(
        f"Trainable scope: {scope}; trainable params: {trainable_params:,} / "
        f"{total_params:,} ({ratio:.4f}%)"
    )


def apply_trainable_scope(model, trainable_scope: str):
    if trainable_scope not in VALID_TRAINABLE_SCOPES:
        raise ValueError(
            f"Unknown trainable_scope={trainable_scope}. "
            f"Choose from {sorted(VALID_TRAINABLE_SCOPES)}."
        )

    if trainable_scope == "default":
        print_trainable_parameter_summary(model, trainable_scope)
        return

    if not hasattr(model, "stop_progress_head"):
        raise ValueError(
            f"trainable_scope={trainable_scope} requires add_stop_progress_head=True "
            "or a checkpoint that already contains stop_progress_head."
        )

    for _, p in model.named_parameters():
        p.requires_grad = False

    use_merger = trainable_scope in {"stop_head_merger", "stop_head_merger_lora"}
    use_lora = trainable_scope in {"stop_head_lora", "stop_head_merger_lora"}

    for name, p in model.named_parameters():
        trainable = name.startswith("stop_progress_head.")
        if use_merger and _is_merger_param(name):
            trainable = True
        if use_lora and _is_lora_param(name):
            trainable = True
        p.requires_grad = trainable

    print_trainable_parameter_summary(model, trainable_scope)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if model_args.add_stop_progress_head and data_args.data_flatten:
        raise ValueError(
            "add_stop_progress_head=True is incompatible with data_flatten=True: "
            "packed sequence training cannot align per-sample stop_progress targets "
            "with sample boundaries."
        )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    is_qwen25_vl = (
        "qwen2.5" in model_args.model_name_or_path.lower()
        or "qwen2_5_vl" in model_type
        or "qwen2_5_vl" in architectures
        or "qwen2.5" in architectures
    )

    if is_qwen25_vl:
        from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
        setattr(config, "lam", model_args.lam)
        setattr(config, "add_stop_progress_head", model_args.add_stop_progress_head)
        setattr(config, "stop_head_hidden_dim", model_args.stop_head_hidden_dim)
        setattr(config, "stop_loss_weight", model_args.stop_loss_weight)
        model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            vggt_model_path=model_args.vggt_model_path
        )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    apply_trainable_scope(model, model_args.trainable_scope)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    save_stop_progress_head(trainer.model, training_args.output_dir)
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    if os.environ.get("JANUSVLN_SAVE_FULL_MODEL", "1") == "0":
        rank0_print("Skipping full model save because JANUSVLN_SAVE_FULL_MODEL=0.")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
