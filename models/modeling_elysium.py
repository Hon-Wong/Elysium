from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel
)
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast

from .build_vit import build_projector, create_clip_vit
from .rantselector import build_adapter

logging.set_verbosity_info()  # Turn on this for debug mode
logger = logging.get_logger(__name__)


DTYPE_MAPPING = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "fp32": torch.float32,
}

# image level
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# video level
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"


class ElysiumConfig(PretrainedConfig):
    model_type = "elysium"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        use_flash_attention: bool = False,
        gradient_checkpointing_enable: bool = False,
        torch_dtype: str = "bf16",
        llm_config: Dict = None,
        visual_config: Dict = None,
        adapter_config: Dict = None,
        projector_config: Dict = None,
        **kwargs,
    ):
        self.use_flash_attention = use_flash_attention
        self.gradient_checkpointing_enable = gradient_checkpointing_enable
        self.torch_dtype = torch_dtype
        self.llm_config = llm_config
        self.visual_config = visual_config
        self.adapter_config = adapter_config
        self.projector_config = projector_config

        super().__init__(**kwargs)


class ElysiumForCausalLM(PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig = ElysiumConfig()):
        super().__init__(config)

        # setup llm
        self.flash_attn_monkey_patch()
        self._setup_llm()

        self._setup_visual_encoder()
        self._setup_adapter()
        self._setup_projector()

        if self.config.torch_dtype:
            logger.info(f"Converting model to {DTYPE_MAPPING[self.torch_dtype]}.")
            self.to(DTYPE_MAPPING[self.torch_dtype])
            logger.info("Done.")

        if self.config.gradient_checkpointing_enable:
            self._enable_gradient_checkpointing()

    def flash_attn_monkey_patch(self):
        if self.config.use_flash_attention:
            # use flash attention
            from .llama_flash_attn_monkey_patch import (
                replace_llama_attn_with_flash_attn,
            )
            logger.info("Flash attention is availiable, patching.")
            replace_llama_attn_with_flash_attn()

    def _setup_visual_encoder(self):
        self.visual_encoder = create_clip_vit(**self.config.visual_config)
        if self.config.visual_config["freeze_vit"]:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = lambda self, mode=True: self
            logger.info("freeze vision encoder")

    def _setup_llm(self):
        # text encoder & load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_config["pretrained_model_name_or_path"])
        self.llm = AutoModelForCausalLM.from_pretrained(self.config.llm_config["pretrained_model_name_or_path"])

        # freeze llm if needed
        if self.config.llm_config["freeze_llm"]:
            for _, param in self.llm.named_parameters():
                param.requires_grad = False
            logger.info("freeze llm")
        
        if hasattr(self.llm.config, "hidden_size"):
            self.config.hidden_size = self.llm.config.hidden_size
        if hasattr(self.llm.config, "hidden_sizes"):
            self.config.hidden_sizes = self.llm.config.hidden_sizes

    def _setup_adapter(self):
        self.adapter = build_adapter(self.config.adapter_config)
        if self.config.adapter_config["freeze_adapter"]:
            for _, param in self.adapter.named_parameters():
                param.requires_grad = False
            self.adapter = self.adapter.eval()
            self.adapter.train = lambda self, mode=True: self
            logger.info("freeze adapter")

    def _setup_projector(self):
        self.llm_proj = build_projector(
            output_hidden_size=self.llm.config.hidden_size,
            input_hidden_size=self.adapter.hidden_size,
            **self.config.projector_config,
        )

    def _encode_vision(self, images, n_frames):
        image_embeds = self.visual_encoder(images)
        adapter_out = self.adapter(image_embeds, n_frames=n_frames)
        vision_embeds = [self.llm_proj(feature) for feature in adapter_out]

        attention_mask = [
            torch.ones(feature.size()[:-1], dtype=torch.long).to(feature.device) for feature in vision_embeds]
        vision_targets = [
            torch.ones(feature.size(), dtype=torch.long).to(feature.device).fill_(-100) for feature in attention_mask]
        return vision_embeds, attention_mask, vision_targets

    def _concat_embedding(self, vision_encode_out, input_ids, attention_mask, labels=None, left_padding=False):
        """ concat vision and text
        """
        vision_embeds, vision_atts, vision_targets = vision_encode_out

        input_embeds = []
        attention_masks = []
        targets = []

        for cur_batch_idx, cur_input_ids in enumerate(input_ids):
            cur_vision_embeds = vision_embeds[cur_batch_idx]
            cur_vision_attn = vision_atts[cur_batch_idx]
            cur_vision_targets = vision_targets[cur_batch_idx]
            cur_attn_masks = attention_mask[cur_batch_idx]

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_image_num = len(image_token_indices)
            image_token_indices = list(image_token_indices) + [cur_input_ids.shape[0]]

            cur_input_embeds = []
            cur_attention_mask = []
            cur_target = []

            # convert text before 1st <image> to embedding
            image_token_index = image_token_indices[0]

            cur_input_embeds.append(
                self.llm.get_input_embeddings()(cur_input_ids[:image_token_index]),
            )
            cur_attention_mask.append(
                cur_attn_masks[:image_token_index],
            )
            if labels is not None:
                cur_target.append(labels[cur_batch_idx, :image_token_index])

            assert cur_image_num == len(cur_vision_embeds), \
                f"Size mismatch! cur_image_num: {cur_image_num}, len(cur_vision_embeds): {len(cur_vision_embeds)}"
            # convert each <image> xxx group into embedding
            for i in range(0, cur_image_num):
                image_token_index = image_token_indices[i]
                cur_input_embeds.append(torch.cat([
                    cur_vision_embeds[i],
                    self.llm.get_input_embeddings()(cur_input_ids[image_token_index+1:image_token_indices[i+1]])
                ]))
                cur_attention_mask.append(torch.cat([
                    cur_vision_attn[i],
                    cur_attn_masks[image_token_index+1:image_token_indices[i+1]]
                ]))
                if labels is not None:
                    cur_target.append(torch.cat([
                        cur_vision_targets[i],
                        labels[cur_batch_idx, image_token_index+1:image_token_indices[i+1]],
                    ]))

            input_embeds.append(torch.cat(cur_input_embeds))
            attention_masks.append(torch.cat(cur_attention_mask))
            if labels is not None:
                targets.append(torch.cat(cur_target))

        # padding
        n_tokens = [embed.shape[0] for embed in input_embeds]
        max_token = max(n_tokens)
        for i in range(len(input_embeds)):
            if max_token > n_tokens[i]:
                self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                pad_token = torch.tensor([self.pad_id] * (max_token - n_tokens[i]))
                pad_embedding = self.llm.get_input_embeddings()(pad_token.to(vision_embeds[i].device))
                pad_attention = torch.zeros(pad_embedding.shape[0], dtype=torch.long).to(vision_embeds[i].device)
                pad_targets = torch.ones(pad_attention.size(), dtype=torch.long).to(vision_embeds[i].device).fill_(-100)

                if left_padding:
                    input_embeds[i] = torch.cat([pad_embedding, input_embeds[i]])
                    attention_masks[i] = torch.cat([pad_attention, attention_masks[i]])
                    if labels is not None:
                        targets[i] = torch.cat([pad_targets, targets[i]])
                else:
                    input_embeds[i] = torch.cat([input_embeds[i], pad_embedding])
                    attention_masks[i] = torch.cat([attention_masks[i], pad_attention])
                    if labels is not None:
                        targets[i] = torch.cat([targets[i], pad_targets])

        inputs_embeds = torch.stack(input_embeds, dim=0).type(self.llm.dtype)
        attention_masks = torch.stack(attention_masks, dim=0)

        if len(targets) > 0:
            targets = torch.stack(targets, dim=0)

        return inputs_embeds, attention_masks, targets

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                frames: torch.LongTensor = None,
                n_frames: List[int] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        # get vision features
        vision_encode_out = self._encode_vision(frames, n_frames)

        inputs_embeds, attention_mask, targets = self._concat_embedding(
            vision_encode_out, input_ids, attention_mask, labels)

        # input to llm
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
        )
        return outputs

    def _enable_gradient_checkpointing(self):
        for model in (self.visual_encoder, self.llm):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def generate(self,
                 frames: torch.Tensor,
                 n_frames: List[int],
                 input_ids: torch.LongTensor,
                 attention_mask: torch.Tensor,
                 **kwargs):
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):
            vision_encode_out = self._encode_vision(frames, n_frames)
            inputs_embeds, attention_mask, _ = self._concat_embedding(
                vision_encode_out, input_ids, attention_mask)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        # parse result text
        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
