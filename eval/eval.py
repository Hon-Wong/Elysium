import copy
import json
import math
import os
import os.path as osp
import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from data.processors.box_processor import BOX_PROCESSORS
from data.video_llm_data import VideoLLMPredictProcessor


global global_box_pool
global_box_pool = {}


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
    beta: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)
    visual_encoder_lr_scale: float = field(default=1.0)


def parse_box_from_raw_text(text, coords_pattern=r"{<(\d+)><(\d+)><(\d+)><(\d+)>}"):
    try:
        raw_coords = re.findall(coords_pattern, text)
        if len(raw_coords) < 1:
            raw_coords = re.findall(r"\[([\d\s,]+)\]", text)
            coords = [[float(coord)/100 for coord in xyxy_str.replace(" ", "").split(",")][:4] for xyxy_str in raw_coords]
            coords = []
            for xyxy_str in raw_coords:
                box = []
                for coord in xyxy_str.replace(" ", "").split(","):
                    box.append(float(coord)/100)
                box = box[:4]
                if len(box) < 4:
                    box = coords[-1]
                    if len(box) < 4:
                        box = [0,0,0,0]
                coords.append(box)
        else:
            coords = [[float(coord) for coord in xyxy_str][:4] for xyxy_str in raw_coords]
        return coords
    except Exception as e:
        print(e)
        return []


class LongVideoDistributedSampler(DistributedSampler):
    def __init__(self, start_indices, **kwargs) -> None:
        self.start_indices = start_indices
        super().__init__(**kwargs)
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        iter_indices = []
        for i in range(self.rank, len(self.start_indices) - 1, self.num_replicas):
            start_index = self.start_indices[i]
            end_index = self.start_indices[i+1]
            iter_indices.extend(indices[start_index:end_index])
        return iter(iter_indices)


class LocalDataset(Dataset):
    BOX_STR_TEMPLATE = "Frame {i}: <box>"
    FRAME_STR_TEMPLATE = "Frame {i}: <image>"

    SOT_QUESTION_TEMPLATE = "{frame_str}This is a video showing an object with coordinates <box> in Frame 1. Please provide the detailed coordinates of the object in each frame."
    RSOT_QUESTION_TEMPLATE = "{frame_str}Please find one {object_class} and provide the detailed coordinates in each frame."
    
    def __init__(self, image_folder, anno_path, clip_len=8, task="RSOT", processor=None):
        self.image_folder = image_folder
        self.processor = processor
        self.clip_len = clip_len
        self.task=task
        self.start_indices = []

        self.box_processor = BOX_PROCESSORS["ours_v1"]
        with open(anno_path, "r") as f:
            anns = [json.loads(line) for line in f]
        self.preprocess(anns)
        
    def preprocess(self, anns):
        self.anns = []
        for item in anns:
            self.start_indices.append(len(self.anns))
            frames_path = item["frames"]
            boxes = item["box"]
            assert len(boxes) == len(frames_path)

            n_clip = math.ceil(len(frames_path) / (self.clip_len - 1))
            for clip_id in range(n_clip):
                clip_frames_path = frames_path[
                    clip_id * (self.clip_len - 1): clip_id * (self.clip_len - 1) + self.clip_len]
                clip_boxes = boxes[
                    clip_id * (self.clip_len - 1): clip_id * (self.clip_len - 1) + self.clip_len]
                clip_data = dict(
                    frames_path=clip_frames_path,
                    box=clip_boxes,
                    inital_box=clip_boxes[0],
                    frame_size=item["frame_size"],
                    object_description=item["object_description"],
                    object_class=item["object_class"],
                    video_folder=item["vid"],
                    seq_id=item["vid"],
                    clip_id=clip_id,
                )
                self.anns.append(clip_data)
        self.start_indices.append(len(self.anns))

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])
        seq_id = item["seq_id"]
        clip_id = item["clip_id"]
        if clip_id == 0:
            initial_box = item['inital_box']
        else:
            initial_box = global_box_pool[f"{seq_id}|{clip_id-1}"]

        frame_len = len(item["frames_path"])
        frame_str = ", ".join(self.FRAME_STR_TEMPLATE.format(**{"i": i + 1}) for i in range(0, frame_len)) + "\n"
        box_str = ", ".join(self.BOX_STR_TEMPLATE.format(**{"i": i + 1}) for i in range(0, frame_len))

        if self.task == "SOT":
            question = self.SOT_QUESTION_TEMPLATE.format(**{"frame_str": frame_str})
            answer = box_str
        elif self.task == "RSOT":
            question = self.RSOT_QUESTION_TEMPLATE.format(**{"frame_str": frame_str, "object_class": item["object_class"]})
            answer = box_str

        if self.box_processor.box_token in question:
            if question.count(self.box_processor.box_token) == 1:
                question = self.box_processor(question, [initial_box])
            else:
                question = self.box_processor(question, item['box'])

        if self.box_processor.box_token in answer:
            if answer.count(self.box_processor.box_token) == 1:
                answer = self.box_processor(answer, [item['inital_box']])
            else:
                answer = self.box_processor(answer, item['box'])

        messages = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]

        data_dict = {
            "id": f"{item['seq_id']}|{item['clip_id']}",
            "vid": f"{item['seq_id']}|{item['clip_id']}",
            "frames": item["frames_path"],
            "image_folder": self.image_folder,
            "question": question,
            "gt": answer,
            "vqa": messages,
            "image_size": item["frame_size"]
        }
        output = self.processor.transform(data_dict)
        return output


class VideoLLMEvaluator:
    def __init__(self, model, data_args, task, **kwargs):
        super().__init__(**kwargs)
        self.data_args = edict(data_args.data)
        self.task = task
        self.dataloader = self.get_dataloader(self.data_args.predict)
        self.model = model.cuda().eval()

    def get_dataloader(self, config):
        df_config = config.data_fetch
        dp_config = config.data_preprocess
        dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
        processor = VideoLLMPredictProcessor(**dp_config)

        dataset = LocalDataset(
            image_folder=df_config.image_folder, 
            anno_path=df_config.anno_path, 
            processor=processor,
            task=self.task
        )

        sampler = LongVideoDistributedSampler(start_indices=dataset.start_indices, dataset=dataset)
        loader = DataLoader(dataset, batch_size=df_config.batch_sizes[0], sampler=sampler, prefetch_factor=None,
                        collate_fn=processor.batch_transform, num_workers=0, shuffle=False)
        return loader

    def predict(self, save_path):
        f = open(save_path, "a")

        generate_params = dict(
            do_sample=False,
            max_new_tokens=2048,
            min_length=4,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        for _, batch in tqdm(enumerate(self.dataloader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            outputs = self.model.generate(
                batch["frames"],
                batch["n_frames"],
                batch["input_ids"],
                batch["attention_mask"],
                **generate_params
            )

            outputs_dict = dict(
                vid=batch["vid"],
                id=batch["id"],
                question=batch["question"],
                prompt=batch["prompt"],
                gt=batch["gt"],
                predict=outputs,
                image_size=batch["image_size"]
            )

            for key in outputs_dict:
                if isinstance(outputs_dict[key], torch.Tensor):
                    outputs_dict[key] = batch[key].cpu().numpy().tolist()
            
            for i, pred in enumerate(outputs):
                pred_boxes = parse_box_from_raw_text(pred)
                global_box_pool[batch["id"][i]] = pred_boxes[-1]

            list_of_dict = [{key: values[i] for key, values in outputs_dict.items()} for i in range(len(list(outputs_dict.values())[0]))]
            
            for line in list_of_dict:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda')

    argument_parser = ArgumentParser()
    argument_parser.add_argument('--config', type=str)
    argument_parser.add_argument('--local_rank', type=int)
    argument_parser.add_argument("--task", type=str, choices=("SOT", "RSOT"), default="SOT", help="specify the task")
    args = argument_parser.parse_args()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)

    model = AutoModelForCausalLM.from_pretrained("elysium_7b", trust_remote_code=True)
    evaluater = VideoLLMEvaluator(model=model, data_args=data_args, task=args.task)

    save_filename = osp.basename(edict(data_args.data).predict.data_fetch.anno_path)
    save_folder = osp.join(training_args.output_dir, "infer_results")
    save_path = osp.join(save_folder, save_filename)
    os.makedirs(save_folder, exist_ok=True)
    evaluater.predict(save_path=save_path)
