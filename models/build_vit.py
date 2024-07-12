import re

from transformers import CLIPVisionModel
import torch
import torch.nn as nn


class ClipVisionTransformer(CLIPVisionModel):
    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_forward_outs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA


def create_clip_vit(
    precision="fp16", pretrained_model_name_or_path: str = "", low_cpu_mem_usage=False, **kwargs
):
    dtype = torch.float16 if "16" in precision else torch.float32
    model = ClipVisionTransformer.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        ignore_mismatched_sizes=True,
    ).cuda()
    return model


def build_projector(
    type: str = "linear", input_hidden_size: int = 1024, output_hidden_size: int = 1024
):
    """build vision projector
    Args:
        type: projector type (linear, mlp2x_gelu, identity)
        input_hidden_size: input hidden size from adaptor
        output_hidden_size: output hidden size to llm
    Returns:
        vision projector module(nn.Module)
    """

    if type == "linear":
        return nn.Linear(input_hidden_size, output_hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(input_hidden_size, output_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*modules)

    if type == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown projector type: {type}")
