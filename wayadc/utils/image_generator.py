import os
import pickle
import random
import re

from keras.applications import xception
import numpy as np
from PIL import Image

from wayadc.utils import helpers


def pre_process_diagnosis_name(diagnosis_name):
    diagnosis_name = re.sub(r'\(.*?\)', '', diagnosis_name).strip()

    tmp = diagnosis_name.lower()

    tmp = re.sub(r'[-/ \n]', r'_', tmp)
    tmp = re.sub(r'[,()\']', r'', tmp)

    return re.sub(r'_+', ' ', tmp)


class ImageGenerator(object):
    def __init__(self, data_dirs, target_size, transformation_pipeline=None):
        self.target_size = target_size
        self.transformation_pipeline = transformation_pipeline

        self.index = []
        self.label_sizes = {}

        data_directory = os.path.join(list(data_dirs)[0], os.pardir)

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

        self.identity_matrix_labels = np.eye(len(self._labels))
        self.identity_matrix_groups = np.eye(len(self._groups))

        for data_dir in data_dirs:
            if isinstance(data_dir, tuple):
                print(data_dir[0])
                for image_file_path, class_index, group_index in data_dir[1]:
                    self.label_sizes[group_index] = self.label_sizes.get(group_index, 0) + 1
                    self.index.append((image_file_path, class_index, group_index))
                continue

            image_details_file_path = os.path.join(data_dir, 'image_details.pickle')

            with open(image_details_file_path, 'rb') as handle:
                image_details = pickle.load(handle)

            nb_discarded = 0

            for image_file_name in image_details:
                if image_details.get(image_file_name).get('in_dataset') or image_details.get(image_file_name).get('is_duplicate'):
                    nb_discarded += 1
                    continue

                diagnosis = image_details.get(image_file_name).get('diagnosis')
                diagnosis = pre_process_diagnosis_name(diagnosis)

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
                            with Image.open(image_file_path) as _:
                                pass
                        except Exception:
                            nb_discarded += 1
                            break

                        self.label_sizes[group_index] = self.label_sizes.get(group_index, 0) + 1
                        self.index.append((image_file_path, class_index, group_index))

            print('Discarded: {}.'.format(nb_discarded))

        for group in self._groups:
            group_index = self._groups.index(group)
            self.label_sizes[group_index] = self.label_sizes.get(group_index, 1)

        print('Found {} images belonging to {} labels and {} groups.'.format(len(self.index), len(self._labels), len(self._groups)))

    def image_generator(self, batch_size, single_epoch=False):
        index = self.index

        def epoch():
            for batch in range(len(index) // batch_size):
                image_batch = []
                label_batch = []
                image_details = []

                for i in range(batch_size):
                    idx = batch * batch_size + i
                    image_file_path, label, group = self.index[idx]

                    im, group = self.__getitem__(idx)
                    image_batch.append(im)
                    label_batch.append(group)
                    image_details.append((image_file_path, label, group))

                yield np.asarray(image_batch), np.asarray(label_batch), image_details

        while True:
            random.shuffle(index)
            yield from epoch()
            if single_epoch:
                raise StopIteration

    def reset(self, data_dirs):
        self.__init__(data_dirs, self.target_size)

    def pre_process_image(self, im):
        im = im.resize(self.target_size, resample=Image.LANCZOS)
        im = np.asarray(im, dtype=np.float32)

        return xception.preprocess_input(im.copy())

    def __getitem__(self, index):
        image_file_path, label, group = self.index[index]

        im = Image.open(image_file_path)
        im = im.convert('RGB')
        im = self.transformation_pipeline(im)

        return im, group

    def __len__(self):
        return len(self.index)
