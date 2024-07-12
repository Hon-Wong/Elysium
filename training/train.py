import copy
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from argparse import ArgumentParser
import torch
import torch.nn as nn
import transformers
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled

from data.video_llm_data import VideoLLMProcessor
from models.modeling_elysium import ElysiumForCausalLM, ElysiumConfig


@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)


@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    visual_encoder_lr_scale: int = field(default=1.0)
    remove_unused_columns: bool = field(default=False)
    using_torch_lr: bool = field(default=False)
    lr_type: str = field(default="")


class LocalDataset(Dataset):
    def __init__(self, data_paths, multi_round_qa=True, processor=None):
        self.anns = []
        for data_path in data_paths:
            image_folder = data_path["image_folder"]
            anno_path = data_path["anno_path"]
            f = open(anno_path, "r")
            if multi_round_qa:
                for line in f:
                    item = json.loads(line)
                    item["image_folder"] = image_folder
                    self.anns.append(item)
            else:
                for line in f:
                    line = json.loads(line)
                    single_round_lines = []
                    vqas = line["vqa"]
                    num_rounds = len(vqas) // 2
                    line.pop("vqa")
                    for i in range(num_rounds):
                        single_vqa = vqas[2 * i: 2 * i + 2]
                        single_round_line = copy.deepcopy(line)
                        single_round_line["vqa"] = single_vqa
                        single_round_line["image_folder"] = image_folder
                        single_round_lines.append(single_round_line)
                    self.anns.extend(single_round_lines)

        self.processor = processor

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):  
        item = copy.deepcopy(self.anns[idx])
        output = self.processor.transform(item)
        return output


class VideoLLMTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def create_optimizer(self):

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            scale_lr_parameters = [p for n, p in opt_model.named_parameters() if (n.startswith("visual_encoder") and p.requires_grad)]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and not n.startswith("visual_encoder") and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and not n.startswith("visual_encoder") and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": scale_lr_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.visual_encoder_lr_scale * self.args.learning_rate
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        print(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

        return self.optimizer


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    if trainer.deepspeed:
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
        trainer._save(output_dir, state_dict=cpu_state_dict)


if __name__ == "__main__":
    global local_rank
    os.environ["WANDB_PROJECT"] = "Elysium"

    argument_parser = ArgumentParser()
    argument_parser.add_argument('--config', type=str)
    argument_parser.add_argument('--local_rank', type=int)
    args = argument_parser.parse_args()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)
    local_rank = args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

    df_config = edict(data_args.data).train.data_fetch
    dp_config = edict(data_args.data).train.data_preprocess
    dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
    processor = VideoLLMProcessor(**dp_config)
    
    train_dataset = LocalDataset(
        data_paths=df_config.data_paths,
        multi_round_qa=df_config.get("multi_round_qa", True),
        processor=processor
    )

    config = ElysiumConfig.from_pretrained("models", trust_remote_code=True)
    model = ElysiumForCausalLM(config)
    print(
        f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = VideoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.batch_transform
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )