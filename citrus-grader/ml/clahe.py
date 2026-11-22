import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        enhanced = self._clahe.apply(lightness)
        result = cv2.merge((enhanced, a_channel, b_channel))
        rgb_result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_result)
