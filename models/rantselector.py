import torch
import torch.nn as nn
import torch.nn.functional as F


class RanTSelecor(nn.Module):
    def __init__(
        self,
        num_features=1408,
        in_token_num=576,
        out_token_num=144,
        min_out_token_num=16,
        max_video_tokens=2048,
        fix_random=False,
        **kwargs,
    ):
        super().__init__()
        self.in_token_num = in_token_num
        self.out_token_num = out_token_num
        self.min_out_token_num = min_out_token_num
        self.max_video_tokens = max_video_tokens
        self.pre_proj = nn.Linear(num_features, 1)
        self.score_proj = nn.Linear(in_token_num, out_token_num)

        self.trans_weight = nn.Parameter(
            torch.zeros(in_token_num, in_token_num), requires_grad=True
        )
        self.trans_bias = nn.Parameter(self._init_bias(), requires_grad=True)

        self.hidden_size = num_features
        self._init_bias()
        self.fix_random = fix_random

    def _init_bias(self):
        tensor = torch.zeros(self.in_token_num)
        m = self.out_token_num
        n = tensor.numel()
        if m > n:
            m = n
        count = 0

        n_group = n // m
        interval = max(4, n_group)

        for j in range(0, interval):
            for i in range(j, n, interval):
                tensor[i] = 0.02
                count += 1
                if count >= m:
                    break
            if count >= m:
                break
        return tensor

    def forward(self, image_embeds, n_frames, noise_epsilon=0.001):
        image_embeds_list = image_embeds.split(n_frames, dim=0)
        # Compute temporal tokens as the mean along the time axis
        ret_tokens = []
        for image_embeds_per_video in image_embeds_list:
            video_raw_token = image_embeds_per_video

            video_raw_token_trans = self.pre_proj(video_raw_token).mT
            video_token_logits = (
                video_raw_token_trans @ self.trans_weight + self.trans_bias
            )

            video_token_logits = video_token_logits.squeeze(1)

            video_token_scores = F.softmax(video_token_logits)
            topk_indices = torch.argsort(video_token_scores, descending=True)[:, :self.out_token_num]
            topk_indices, _ = torch.sort(topk_indices)
            video_topk_token = video_raw_token[
                torch.arange(video_raw_token.size(0)).unsqueeze(1), topk_indices
            ]
            ret_tokens.append(video_topk_token)
        return ret_tokens


def build_adapter(config):
    return RanTSelecor(**config)
