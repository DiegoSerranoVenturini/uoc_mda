import pickle
import numpy as np

from advdatamining.cfg.constants import FilePathConstants


def load_pickle(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


class ImageReader:

    @staticmethod
    def read_images():
        return load_pickle(FilePathConstants.IMAGE_PICKLE)
