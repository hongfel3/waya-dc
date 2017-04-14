import os
import pickle
import random

import numpy as np
from PIL import Image

from wdata.scrape import scrape_dermweb
from wayadc.utils import helpers


class ImageGenerator(object):
    def __init__(self, data_dirs, valid_split=None):
        self.valid_split = valid_split

        self.index = []
        self.label_sizes = {}

        # sets do not support indexing
        for data_dir in data_dirs:
            data_directory = os.path.join(data_dir, os.pardir)
            break

        diagnosis_to_cluster_file_path = os.path.join(data_directory, 'diagnoses_to_cluster.pickle')
        with open(diagnosis_to_cluster_file_path, 'rb') as handle:
            self.diagnosis_to_cluster = pickle.load(handle)

        cluster_to_group_file_path = os.path.join(data_directory, 'clusters_to_group.pickle')
        with open(cluster_to_group_file_path, 'rb') as handle:
            self.cluster_to_group = pickle.load(handle)

        labels = set()
        for diagnosis, cluster in self.diagnosis_to_cluster.items():
            labels.add(cluster)
        self._labels = sorted(list(labels))

        groups = set()
        for cluster, group in self.cluster_to_group.items():
            groups.add(group)
        self._groups = sorted(list(groups))

        for data_dir in data_dirs:
            image_files = helpers.list_dir(data_dir, images_only=True)
            image_details_file_path = os.path.join(data_dir, 'image_details.pickle')

            with open(image_details_file_path, 'rb') as handle:
                image_details = pickle.load(handle)

            nb_discarded = 0

            for image_file_name in image_details:
                if image_details.get(image_file_name).get('in_dataset') or image_details.get(image_file_name).get('is_duplicate'):
                    nb_discarded += 1
                    continue

                diagnosis = image_details.get(image_file_name).get('diagnosis')
                # to make sure this is consistent across data sets
                diagnosis = scrape_dermweb.pre_process_diagnosis_name(diagnosis)

                cluster = self.diagnosis_to_cluster.get(diagnosis)
                group = self.cluster_to_group.get(cluster)

                try:
                    class_index = self._labels.index(cluster)
                except ValueError:
                    nb_discarded += 1
                    continue

                try:
                    group_index = self._groups.index(group)
                except ValueError:
                    nb_discarded += 1
                    continue

                # by far the fastest way to get the file path when we don't have the extension
                for ext in helpers.image_ext_whitelist:
                    image_file_path = os.path.join(data_dir, '{}.{}'.format(image_file_name, ext))
                    if os.path.isfile(image_file_path):
                        # make sure this is a valid image file
                        try:
                            im = Image.open(image_file_path)
                            self.label_sizes[class_index] = self.label_sizes.get(class_index, 0) + 1
                            self.index.append((image_file_path, class_index, group_index))
                        except Exception:
                            nb_discarded += 1
                            break

            print('Discarded: {}.'.format(nb_discarded))

        print('Found {} images belonging to {} labels and {} groups.'.format(len(self.index), len(self._labels), len(self._groups)))

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
                        image_file_path, label, group = index[batch * batch_size + i]
                    except IndexError:
                        return

                    im = Image.open(image_file_path)
                    im = im.convert('RGB')
                    im = im.resize(target_size, resample=Image.LANCZOS)
                    im = np.asarray(im, dtype=np.float32)

                    if pre_processing_function:
                        im = pre_processing_function(im.copy())

                    image_batch.append(im)
                    label_batch.append(identity_matrix[group])

                yield np.asarray(image_batch), np.asarray(label_batch)

        while True:
            identity_matrix = np.eye(len(self._groups))
            random.shuffle(index)

            yield from epoch()

    def reset(self, data_dirs):
        self.__init__(data_dirs)
