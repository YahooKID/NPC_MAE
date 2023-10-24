import math
import numbers
import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import (adjust_brightness, adjust_color, adjust_contrast,
                        adjust_hue)
from mmcv.transforms import BaseTransform
from PIL import Image, ImageFilter
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
import copy


@TRANSFORMS.register_module()
class SegRandomResizedCrop(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        size (Sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (Tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (Tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size: Union[int, Sequence[int]],
                 scale: Tuple = (0.08, 1.0),
                 ratio: Tuple = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        super().__init__()
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img: np.ndarray,
                   scale: Tuple,
                   ratio: Tuple,
                   max_attempts: int = 10) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (Tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (Tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    def transform(self, results: dict) -> dict:
        """Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        """
        img = results['img']
        gt_seg_map = results['gt_seg_map']
        get_params_args = dict(
            img=img,
            scale=self.scale,
            ratio=self.ratio,
            max_attempts=self.max_attempts)
        ymin, xmin, ymax, xmax = self.get_params(**get_params_args)
        # img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
        results['img'] = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        
        results['gt_seg_map'] = mmcv.imresize(
            gt_seg_map,
            tuple(self.size[::-1]),
            interpolation='nearest',
            backend=self.backend)
        results['gt_seg_map'][results['gt_seg_map']==2.] = 1. 
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str
    
@TRANSFORMS.register_module()
class SegRandomResizedCropVal(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        size (Sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (Tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (Tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size: Union[int, Sequence[int]],
                 scale: Tuple = (0.08, 1.0),
                 ratio: Tuple = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        super().__init__()
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img: np.ndarray,
                   scale: Tuple,
                   ratio: Tuple,
                   max_attempts: int = 10) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (Tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (Tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    def transform(self, results: dict) -> dict:
        """Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        """
        img = results['img']
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
        get_params_args = dict(
            img=img,
            scale=self.scale,
            ratio=self.ratio,
            max_attempts=self.max_attempts)
        ymin, xmin, ymax, xmax = self.get_params(**get_params_args)
        # img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
        results['img'] = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        
        # results['gt_seg_map'] = mmcv.imresize(
        #     gt_seg_map,
        #     tuple(self.size[::-1]),
        #     interpolation='nearest',
        #     backend=self.backend)
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = copy.deepcopy(gt_seg_map)
            results['gt_seg_map'][results['gt_seg_map']==2.] = 1. 
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str
