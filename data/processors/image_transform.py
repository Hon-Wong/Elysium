import numpy as np
from PIL import Image


class PILToNdarray:
    def __init__(self,):
        pass

    def __call__(self, image: Image.Image):
        image_array = np.array(image)
        return image_array.astype(np.float32)


class Rescale:
    def __init__(self, rescale_factor: float):
        self.scale = rescale_factor

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError
        return image * self.scale
