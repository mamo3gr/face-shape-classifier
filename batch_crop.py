import os
from argparse import ArgumentParser
from logging import getLogger, basicConfig, INFO

import cv2.cv2 as cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from mtcnn.exceptions.invalid_image import InvalidImage

parser = ArgumentParser(description='batch cropping face region')
parser.add_argument('root_dir_in', help='input root directory')
parser.add_argument('root_dir_out', help='output root directory')
args = parser.parse_args()

# workaround for avoiding error that 'cuDNN failed to initialize'
#   ref:
#     - https://qiita.com/hirotow/items/6233266b5fe970203ec5
#     - https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:',
              tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


class FaceNotFound(Exception):
    """No face is found in the input image"""


class FaceRegionExtractor:
    def __init__(self, logger=None):
        self._logger = logger or getLogger(__name__)
        self._detector = MTCNN()

    def cutout_face_region(self, img: np.ndarray) -> np.ndarray:
        try:
            res = self._detector.detect_faces(img)
        except InvalidImage as e:
            self._logger.warning(f'{e}')
            raise InvalidImage(f'{e}')

        if not res:
            raise FaceNotFound('face not found')

        x, y, w, h = res[0].get('box')
        # clamp indices
        img_width, img_height, *ignored = img.shape
        y_start = np.clip(y, 0, img_height)
        y_end = np.clip(y + h, 0, img_height)
        x_start = np.clip(x, 0, img_width)
        x_end = np.clip(x + w, 0, img_width)

        img_face = img[y_start:y_end, x_start:x_end, :]
        return img_face


def crop_face_region_batch(root_dir_in: str, root_dir_out: str, logger=None):
    logger = logger or getLogger('crop_face_region_batch')
    extractor = FaceRegionExtractor()

    for dirpath, _, filenames in os.walk(root_dir_in):
        for filename in filenames:
            file_in = os.path.join(dirpath, filename)
            img = cv2.imread(file_in, cv2.IMREAD_COLOR)
            logger.info(f'loaded file: {file_in}')

            if img is None:  # error while loading
                logger.warning(f'failed to load file: {file_in}')
                continue

            file_out = os.path.join(
                root_dir_out,
                os.path.relpath(file_in, start=root_dir_in)
            )
            if os.path.exists(file_out):
                logger.info(f'output file exists. skipped: {file_out}')
                continue

            try:
                img_cropped = extractor.cutout_face_region(img)
            except (InvalidImage, FaceNotFound) as e:
                logger.warning(f'{e}')
                continue

            dir_out = os.path.dirname(file_out)
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)
                logger.info(f'make directory: {dir_out}')

            cv2.imwrite(file_out, img_cropped)
            logger.info(f'write file: {file_out}')


def main():
    basicConfig(level=INFO)
    logger = getLogger(__name__)

    root_dir_in = args.root_dir_in
    root_dir_out = args.root_dir_out
    crop_face_region_batch(root_dir_in, root_dir_out, logger)


if __name__ == '__main__':
    main()
