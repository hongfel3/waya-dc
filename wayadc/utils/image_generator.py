import os
import pickle
import random

import numpy as np
from PIL import Image

from wayadc.utils import helpers


class ImageGenerator(object):
    def __init__(self, data_dirs, labels, valid_split=None):
        self._labels = labels
        self.valid_split = valid_split

        self.index = []
        self.label_sizes = {}

        for data_dir in data_dirs:
            image_files = helpers.list_dir(data_dir, images_only=True)
            image_details_file_path = os.path.join(data_dir, 'image_details.pickle')

            with open(image_details_file_path, 'rb') as handle:
                image_details = pickle.load(handle)

            for image_file_name in image_details:
                diagnosis = image_details.get(image_file_name).get('parent_diagnosis')

                try:
                    class_index = self._labels.index(diagnosis)
                except ValueError:
                    continue

                for image_file in image_files:
                    if image_file_name in image_file:
                        image_file_path = os.path.join(data_dir, image_file)

                        self.label_sizes[class_index] = self.label_sizes.get(class_index, 0) + 1
                        self.index.append((image_file_path, class_index))

                        image_files.remove(image_file)
                        break

        print('Found {} images belonging to {} labels.'.format(len(self.index), len(self._labels)))

        if self.valid_split:
            x = int(len(self.index) * valid_split)
            random.shuffle(self.index)

            self.valid_index = self.index[:x]
            self.index = self.index[x:]

    def image_generator(self, batch_size, target_size, pre_processing_function=None, valid=False):
        index = self.valid_index if valid else self.index

        def epoch():
            for batch in range(len(index) // batch_size):
                image_batch = []
                label_batch = []

                for i in range(batch_size):
                    try:
                        image_file_path, label = index[batch * batch_size + i]
                    except IndexError:
                        return

                    im = Image.open(image_file_path)
                    im = im.convert('RGB')
                    im = im.resize(target_size, resample=Image.LANCZOS)
                    im = np.asarray(im, dtype=np.float32)

                    if pre_processing_function:
                        im = pre_processing_function(im.copy())

                    image_batch.append(im)
                    label_batch.append(identity_matrix[label])

                yield np.asarray(image_batch), np.asarray(label_batch)

        while True:
            identity_matrix = np.eye(len(self._labels))
            random.shuffle(index)

            yield from epoch()

    def reset(self, data_dirs):
        self.__init__(data_dirs, self._labels)
