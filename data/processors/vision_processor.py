import torchvision.transforms as T
from torchvision.transforms import Compose

import data.processors.image_transform as eT


def create_transform(ops_cfg):
    transform_list = []
    for op in ops_cfg:
        kwargs = ops_cfg[op]
        if hasattr(T, op):
            transform_list.append(getattr(T, op)(**kwargs))
        elif hasattr(eT, op):
            transform_list.append(getattr(eT, op)(**kwargs))
        else:
            raise RuntimeError(f'no op {op} in torchvision.transforms and data.processors.image_transform')

    return Compose(transform_list)


class VisionProcessor:
    def __init__(self, ops):
        self.transform = create_transform(ops)

    def __call__(self, data):
        if isinstance(data, list):
            return [self.transform(_data) for _data in data]
        return self.transform(data)


if __name__ == "__main__":
    ops = {
        "PILToNdarray": {},
        "Resize": {
            "size": [336, 336],
            "interpolation": 3
        },
        "Rescale": {
            "rescale_factor": 0.333
        },
        "ToTensor": {},
        "Normalize": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711]
        }
    }
    vp = VisionProcessor(ops)
