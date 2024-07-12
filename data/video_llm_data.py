import json
import os
import random
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from data.processors.online_vqa_processor import OnlineVQAProcessor
from data.processors.vision_processor import VisionProcessor
from data.processors.vqa_processor import VQAProcessor
from utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IMAGE_TOKEN_INDEX


def load_local_frame(frame_path):
    return Image.open(frame_path).convert("RGB"), frame_path


def random_index(frames_length, num_segments, average=False):
    if frames_length <= num_segments:
        return [i for i in range(frames_length)] + (num_segments - frames_length) * [
            frames_length - 1
        ]
    else:
        result = []
        stride = frames_length // num_segments
        s_list = [stride] * num_segments
        for i in range(frames_length - num_segments * stride):
            s_list[i] += 1
        if not average:
            random.shuffle(s_list)
        cursor = 0
        for idx, each_stride in enumerate(s_list):
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if not average:
                result.append(random.randint(left, right - 1))
            else:
                result.append(left)
        return result


DEFAULT_PROMPT_TEMPLATE = ""
DEFAULT_LABEL_PROMPT = ""


class VideoLLMProcessor(object):
    def __init__(
        self,
        prompt_keys: List[str] = None,
        frames_key: str = "frames",
        frames_ops: Any = None,
        label_key: Union[str, List] = None,
        meta_keys: List[str] = ["vid", "source"],
        key_mapping: dict = None,
        padding_side: str = "right",
        tokenizer: str = "",
        trust_remote_code: bool = False,
        eos_token: str = None,
        max_seq_len: int = 512,
        max_prompt_len: int = 512,
        input_prompt_template: Union[str, dict] = DEFAULT_PROMPT_TEMPLATE,
        label_prompt: Union[str, dict] = DEFAULT_LABEL_PROMPT,
        with_visual: bool = True,
        sample_method: str = "global_random",
        max_frames: int = 1000000,
        max_batch_frames: int = 16,
        num_segments: int = 8,
        training: bool = True,
        verbose: bool = True,
        task_type: str = "vqa",
        shuffle_vqa: bool = False,
        vqa_processor_params: dict = {},
        online_vqa_processor_params: dict = {},
        timestamp_params: dict = {},
        clip_frames: list = [4, 32],
        clip_interval: list = [1, 60],
        extra_sample_keys: list = ["box"],
        truncate_mode: str = "qa",
    ):
        # vision data attributes
        self.with_visual = with_visual
        self.frames_key = frames_key
        self.num_segments = num_segments
        self.max_frames = max_frames
        self.max_batch_frames = max_batch_frames
        self.extra_sample_keys = extra_sample_keys
        self.truncate_mode = truncate_mode

        # vision processors
        if self.with_visual:
            if isinstance(frames_ops, str):
                self.video_processor = AutoImageProcessor.from_pretrained(frames_ops)
            else:
                self.video_processor = VisionProcessor(frames_ops)

        assert padding_side in (
            "left",
            "right",
        ), f"unsupport padding_size: {padding_side}"
        self.padding_side = padding_side

        self.input_prompt_template = input_prompt_template
        self.label_prompt = label_prompt
        self.prompt_keys = prompt_keys or self.get_keys_from_template()
        self.label_key = [label_key] if isinstance(label_key, str) else label_key
        self.meta_keys = meta_keys
        self.key_mapping = key_mapping
        self.training = training
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.sample_method = sample_method
        self.verbose = verbose

        # load tokenizer
        self.eos_token = eos_token
        local_path = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=trust_remote_code
        )
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        self.eos_id = (
            self.tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token
            else self.tokenizer.eos_token_id
        )
        self.ignore_index = -100

        self.task_type = task_type
        self.vision_placeholder =  DEFAULT_IMAGE_TOKEN
        self.vision_placeholder_index = IMAGE_TOKEN_INDEX
            
        self.shuffle_vqa = shuffle_vqa
        self.vqa_processor = VQAProcessor(
            self.label_key[0],
            self.vision_placeholder,
            **vqa_processor_params,
        )
        self.online_vqa_processor = OnlineVQAProcessor(
            self.vision_placeholder,
            **online_vqa_processor_params
        )

        self.timestamp_params = timestamp_params
        self.clip_frames = clip_frames
        self.clip_interval = clip_interval

    def preprocess(self, data_dict):
        for key in data_dict.keys():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = data_dict[key].tolist()

        # frames preprocess
        if self.with_visual:
            data_dict[self.frames_key] = data_dict[self.frames_key][: self.max_frames]
            self.sample_frames(data_dict)
            frames = data_dict[self.frames_key]
            assert isinstance(frames, list)
            if isinstance(frames[0], (str, os.PathLike)):
                frames = [
                    os.path.join(data_dict["image_folder"], frame) for frame in frames
                ]
                with ThreadPoolExecutor(max_workers=32) as executor:
                    frames = [
                        frame[0] for frame in executor.map(load_local_frame, frames)
                    ]
            first_frame = frames[0]
            image_size = first_frame.size
            data_dict["image_size"] = image_size
            num_frames = len(frames)
            data_dict["n_frames"] = num_frames
            data_dict[self.frames_key] = frames

            # set data_mode
            if self.task_type == "vqa":
                vqa_key = self.label_key[0]
                if vqa_key in data_dict:
                    if not isinstance(data_dict[vqa_key], list):
                        data_dict[vqa_key] = json.loads(data_dict[vqa_key])
                    if self.shuffle_vqa and len(data_dict[self.frames_key]) > 1:
                        n_per_group = 2
                        group_indices = list(
                            range(0, len(data_dict[vqa_key]), n_per_group)
                        )
                        random.shuffle(group_indices)
                        tmp = []
                        for i in group_indices:
                            tmp.extend(data_dict[vqa_key][i : i + n_per_group])
                        data_dict[vqa_key] = tmp

            else:
                data_dict["input_prompt_template"] = self.input_prompt_template
                data_dict["label_prompt"] = self.label_prompt

            self.add_vision_placeholders_in_prompt(data_dict)
        else:
            data_dict["input_prompt_template"] = self.input_prompt_template
            data_dict["label_prompt"] = self.label_prompt
            data_dict[self.frames_key] = []
        return data_dict

    def transform(self, data_dict):
        data_dict = self.preprocess(data_dict)

        output = dict()
        # add meta data
        for key in self.meta_keys:
            output[key] = data_dict.get(key, "unknown")

        if self.with_visual:
            if len(data_dict[self.frames_key]) == 0:
                return None
            output[self.frames_key] = self.build_visual(data_dict=data_dict)

        output.update(self.build_text(data_dict=data_dict))
        if "input_ids" in output:
            cur_input_ids = output["input_ids"]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # logger.info("image_token_indices: ", torch.where(cur_input_ids == vision_placeholder_index))
            cur_image_num = len(image_token_indices)
            if cur_image_num != len(data_dict[self.frames_key]):
                print(cur_image_num, len(data_dict[self.frames_key]))
                data_dict.pop(self.frames_key)
                print(data_dict)
                print(output)
                print([index for index in output["input_ids"].cpu().numpy()])
                raise ValueError
        return output

    def collate_frames(self, batch_data, collate_data):
        frames_list = [data.pop("frames") for data in batch_data]
        frame_len_list = [len(frames) for frames in frames_list]

        collate_data["frames"] = torch.cat(frames_list, dim=0)
        collate_data["n_frames"] = frame_len_list

        return batch_data, collate_data

    def padding_sequence(self, inputs, value):
        """Pad input sequence(input_ids, attention_mask, label) to `max_length`,
        fill padding place with `value`
        """
        max_length = max([len(d) for d in inputs])
        padded_data = []
        for t in inputs:
            if len(t) < max_length:
                pad_len = max_length - len(t)
                pad = (0, pad_len) if self.padding_side == "right" else (pad_len, 0)
                t = F.pad(t, pad, value=value)
            padded_data.append(t)
        return torch.stack(padded_data)

    def batch_transform(self, batch_data):
        collate_data = {}

        if self.with_visual:
            batch_data, collate_data = self.collate_frames(batch_data, collate_data)

        # collate all meta keys as list(str)
        all_keys = list(batch_data[0].keys())
        for key in all_keys:
            if isinstance(batch_data[0][key], str) or key in self.meta_keys:
                collate_data[key] = [data.pop(key) for data in batch_data]

        input_ids = [data.pop("input_ids") for data in batch_data]
        input_ids = self.padding_sequence(input_ids, value=self.pad_id)

        attention_mask = [data.pop("attention_mask") for data in batch_data]
        attention_mask = self.padding_sequence(attention_mask, value=0)
        collate_data.update(dict(input_ids=input_ids, attention_mask=attention_mask))

        if "labels" in batch_data[0].keys():
            label = [data.pop("labels") for data in batch_data]
            label = self.padding_sequence(label, value=self.ignore_index)
            collate_data["labels"] = label

        collate_data.update(default_collate(batch_data))
        collate_data["meta_keys"] = self.meta_keys
        collate_data["vision_placeholder_index"] = self.vision_placeholder_index
        collate_data["vision_placeholder"] = self.vision_placeholder
        return collate_data

    def add_vision_placeholders_in_prompt(self, data_dict):
        """For mixture training with video/image datasets, we refine media tokens in prompt.
        - in image mode: replace <video> with [Frame i: <image>] * n_frames
        - in video mode: replace <image> with <video> directly
        """

        def add_timestamp(
            frame_count,
            frame_prefix_pattern="{i}s: ",
            offset=1,
            remove_single_frame_timestamp=True,
            sep="; ",
            remove_last_sep=False,
            end_symbol="\n",
        ):
            if frame_count == 1 and remove_single_frame_timestamp:
                return DEFAULT_IMAGE_TOKEN
            image_mode_prompt = ""
            for i in range(frame_count):
                if "{i}" in frame_prefix_pattern:
                    frame_prefix = frame_prefix_pattern.format(i=i + offset)
                else:
                    frame_prefix = frame_prefix_pattern

                if i == frame_count - 1 and remove_last_sep:
                    image_mode_prompt += frame_prefix + DEFAULT_IMAGE_TOKEN
                    continue
                image_mode_prompt += frame_prefix + DEFAULT_IMAGE_TOKEN + sep
            return image_mode_prompt + end_symbol

        # in image mode, replace <video> with [Frame i: <image>] * n_frames
        if self.task_type == "vqa":
            if self.task_type in data_dict:
                image_mode_prompt = add_timestamp(
                    len(data_dict[self.frames_key]), **self.timestamp_params
                )

                exist_flag = False
                for item in data_dict[self.task_type]:
                    if DEFAULT_VIDEO_TOKEN in item["value"]:
                        exist_flag = True
                        item["value"] = item["value"].replace(
                            DEFAULT_VIDEO_TOKEN, image_mode_prompt
                        )
                    elif DEFAULT_IMAGE_TOKEN in item["value"]:
                        exist_flag = True  # interleaved mode
                        break
                if not exist_flag:
                    data_dict[self.task_type][0]["value"] = (
                        image_mode_prompt + data_dict[self.task_type][0]["value"]
                    )
        else:
            image_mode_prompt = add_timestamp(
                len(data_dict[self.frames_key]), **self.timestamp_params
            )
            data_dict["input_prompt_template"] = data_dict["input_prompt_template"].replace(DEFAULT_VIDEO_TOKEN, image_mode_prompt)

    def sample_frames(self, data_dict):
        # vision inputs
        average_draw = not self.training
        frames = data_dict[self.frames_key]

        if self.sample_method == "global_random":
            # globally sample `self.num_segments` frames
            frames_index = random_index(len(frames), self.num_segments, average_draw)
            part_frames = [frames[i] for i in frames_index]
        elif self.sample_method == "global":
            part_frames = frames
            # if not self.training:
            if len(part_frames) > self.max_batch_frames:
                frames_index = random_index(
                    len(part_frames), self.max_batch_frames, average_draw
                )
                part_frames = [part_frames[i] for i in frames_index]
        elif self.sample_method == "random_clip":
            total_frames = len(data_dict[self.frames_key])
            if self.training:
                clip_len = random.randint(self.clip_frames[0], self.clip_frames[1])
                interval = random.randint(self.clip_interval[0], self.clip_interval[1])
                if clip_len * interval >= total_frames:
                    interval = 1
                    start = 0
                else:
                    start = random.randint(0, total_frames - clip_len * interval)

                part_frames = frames[start : start + clip_len * interval : interval]
                for key in self.extra_sample_keys:
                    assert (
                        len(data_dict[key]) == total_frames
                    ), f"Mismatch between {key}_len: {len(data_dict[key])} and frame_len {total_frames}!"
                    data_dict[key] = data_dict[key][
                        start : start + clip_len * interval : interval
                    ]
            else:
                clip_len = self.clip_frames[-1]
                interval = 1
                start = 0
                part_frames = frames[0 : clip_len * interval : interval]
                for key in self.extra_sample_keys:
                    if key in data_dict:
                        assert (
                            len(data_dict[key]) == total_frames
                        ), f"Mismatch between {key}_len: {len(data_dict[key])} and frame_len {total_frames}!"
                        data_dict[key] = data_dict[key][
                            start : start + clip_len * interval : interval
                        ]

        else:
            raise NotImplementedError()
        data_dict[self.frames_key] = part_frames

    def build_visual(self, data_dict):
        frames = data_dict[self.frames_key]
        if isinstance(self.video_processor, VisionProcessor):
            ret = self.video_processor(frames)
            ret = torch.stack(ret)
        elif isinstance(self.video_processor, BaseImageProcessor):
            ret = self.video_processor(
                [np.asarray(frame.convert("RGB")) for frame in frames],
                return_tensors="pt",
            ).data["pixel_values"]
        else:
            raise NotImplementedError
        return ret

    def get_prompt(self, data_dict):
        text_inputs = OrderedDict()
        for key in self.prompt_keys:
            text_inputs[key] = data_dict[key] or ""

        # format input prompt keys
        input_prompt_template = data_dict["input_prompt_template"]
        if isinstance(input_prompt_template, str):
            prompt = input_prompt_template.format(**text_inputs)
        else:
            raise NotImplementedError()

        if self.with_visual:
            if self.vision_placeholder not in prompt:
                # for visual related tasks, if vision_placeholder not in the prompt,
                # add vision_placeholder <image> <video> before text tokens
                prompt = self.vision_placeholder + prompt
            prompt_ids = self.tokenizer_vision_placeholder(prompt)
        else:
            # for text-only tasks, do tokenizing directly
            prompt_ids = self.tokenizer.encode(prompt)

        # add <sep> special token
        prompt_ids = prompt_ids[: self.max_prompt_len]

        # add label prompt
        _label_prompt = data_dict["label_prompt"]
        if isinstance(_label_prompt, str):
            label_prompt = _label_prompt
        else:
            raise NotImplementedError()

        label_prompt = label_prompt.format(**text_inputs)
        label_prompt_ids = self.tokenizer.encode(label_prompt)

        prompt += label_prompt
        prompt_ids += label_prompt_ids
        return prompt, prompt_ids

    def get_label(self, data_dict):
        label = data_dict[self.label_key[0]]
        response_ids = self.tokenizer.encode(label)
        return label, response_ids

    def build_text(self, data_dict):
        if self.task_type == "vqa":
            if self.label_key[0] in data_dict:
                prompt_list, response_list = self.vqa_processor(data_dict)
            else:
                prompt_list, response_list = self.online_vqa_processor(data_dict)
            prompt_ids_list = [self.tokenizer_vision_placeholder(prompt) for prompt in prompt_list]

            try:
                response_ids_list = [
                    self.tokenizer.encode(response, add_special_tokens=False)
                    for response in response_list
                ]
                input_ids, label_mask = [], []
            except Exception as e:
                raise e

            for prompt_id, response_id in zip(prompt_ids_list, response_ids_list):
                if (len(input_ids) + len(prompt_id) + len(response_id) >= self.max_seq_len - 1
                    and self.truncate_mode == "qa"
                ):  # truncating conversation instead of truncating a sentence
                    print(
                        f"Warning! Get incoming text length {len(input_ids) + len(prompt_id) + len(response_id)} >= max length {self.max_seq_len}. Truncate qa now."
                    )
                    break
                elif (len(input_ids) + len(prompt_id) + len(response_id) >= self.max_seq_len - 1
                    and self.truncate_mode == "text"
                ):
                    print(
                        f"Warning! Get incoming text length {len(input_ids) + len(prompt_id) + len(response_id)} >= max length {self.max_seq_len}. Truncate text now."
                    )
                    input_ids += prompt_id + response_id
                    input_ids = input_ids[: self.max_seq_len - 1]
                    label_mask += [0] * len(prompt_id) + [1] * len(response_id)
                    label_mask = label_mask[: self.max_seq_len - 1]
                    input_ids = input_ids + [self.eos_id]
                    label_mask = label_mask + [1]
                    break
                input_ids += prompt_id + response_id
                label_mask += [0] * len(prompt_id) + [1] * len(response_id)
                input_ids = input_ids + [self.eos_id]
                label_mask = label_mask + [1]

            input_mask = [1] * len(input_ids)
            prompt = " ".join(prompt_list)
            response = " ".join(response_list)
        else:
            prompt, prompt_ids = self.get_prompt(data_dict)
            response, response_ids = self.get_label(data_dict)

            input_mask = [1] * len(prompt_ids)
            label_mask = [0] * len(prompt_ids)
            response_max_len = self.max_seq_len - len(prompt_ids)
            response_ids = response_ids[: response_max_len - 1] + [self.eos_id]
            input_mask += [1] * len(response_ids)
            label_mask += [1] * len(response_ids)

            input_ids = prompt_ids + response_ids

        input_ids = torch.as_tensor(input_ids, dtype=torch.int64)

        label_mask = torch.as_tensor(label_mask, dtype=torch.int64)
        attention_mask = torch.as_tensor(input_mask, dtype=torch.int64)
        label = input_ids.masked_fill(label_mask != 1, self.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "prompt": prompt,
            "gt": response,
        }

    def get_keys_from_template(self):
        def find_var_name(template):
            pattern = re.compile(r"\{(\w+)\}")
            keys = pattern.findall(template)
            return keys

        input_prompt_key = find_var_name(self.input_prompt_template)
        label_prompt_key = find_var_name(self.label_prompt)
        prompt_keys = list(set(input_prompt_key + label_prompt_key))
        return prompt_keys

    def tokenizer_vision_placeholder(self, prompt, add_bos=False):
        def join_lists(*lists, sep):
            result = []
            for i, lst in enumerate(lists):
                if i > 0 and sep:
                    result.extend([sep])
                result.extend(lst)
            return result

        prompt_chunks = [
            self.tokenizer.encode(chunk)
            for chunk in prompt.split(self.vision_placeholder)
        ]
        input_ids = join_lists(*prompt_chunks, sep=self.vision_placeholder_index)
        if add_bos:
            input_ids = [self.tokenizer.bos_token_id] + input_ids

        return input_ids


class VideoLLMPredictProcessor(VideoLLMProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_text(self, data_dict):
        question = ""
        if self.task_type == "vqa":
            if self.label_key[0] in data_dict:
                if isinstance(data_dict[self.label_key[0]], str):
                    vqa_list = json.loads(data_dict[self.label_key[0]])
                else:
                    vqa_list = data_dict[self.label_key[0]]
                if len(vqa_list) == 1:
                    vqa_list.append({"from": "gpt", "value": "Unknown"})
                data_dict[self.label_key[0]] = vqa_list
                question = vqa_list[0]["value"]
                prompt_list, gt_list = self.vqa_processor(data_dict)
            else:
                raise NotImplementedError
            prompt_ids = self.tokenizer_vision_placeholder(prompt_list[0])
            prompt, gt = prompt_list[0], gt_list[0]
            gt = gt.replace("</s>", "")
        else:
            prompt, prompt_ids = self.get_prompt(data_dict)
            gt = data_dict.get(self.label_key[0], "Unknown")

        ret = {
            "input_ids": torch.as_tensor(prompt_ids, dtype=torch.int64),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.int64),
            "prompt": prompt,
            "gt": gt,
            "question": question,
        }

        ret["vid"] = data_dict["vid"]
        # ret["id"] = data_dict["id"]
        ret["image_size"] = data_dict.get("image_size", "unknown")
        return ret
