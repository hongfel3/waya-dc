"""
Image classification using deep convolutional neural networks.

"""

import math
import os
import shutil

import click
from keras import applications
from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
import numpy as np
from sklearn import metrics

from dlutils import plot_image_batch_w_labels
from wayadc.utils import helpers
from wayadc.utils import image_generator


#
# directories and paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')
data_dir = os.path.join(path, 'data')  # data sets are saved in this dir by `$ python3 wayadc/utils/get_datasets.py`

cached_base_model_outputs = os.path.join(cache_dir, 'base_model_outputs_{}.npy')
cached_labels = os.path.join(cache_dir, 'labels_{}.npy')
model_checkpoint = os.path.join(cache_dir, 'model_checkpoint.h5')
valid_image_batch_plot = os.path.join(cache_dir, 'valid_image_batch_plot_{}.png')
tensor_board_logs = os.path.join(cache_dir, 'tensor_board_logs/')


def get_base_model(input_tensor, base_model_name):
    """
    Get a pre-trained base model for transfer learning.

    :param input_tensor: Input tensor to the base model. There are some restrictions on the shape of this tensor
                         depending on which base model is being used.
    :param base_model_name: String corresponding to the desired base model.
    :return: The pre-trained base model.
    """
    if base_model_name == 'vgg16':
        base_model = applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)
    elif base_model_name == 'vgg19':
        base_model = applications.vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)
    elif base_model_name == 'inception_v3':
        base_model = applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)
    elif base_model_name == 'resnet50':
        base_model = applications.resnet50.ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)
    elif base_model_name == 'xception':
        base_model = applications.xception.Xception(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor)
    else:
        assert False, 'Invalid base model name: {}.'.format(base_model_name)

    return base_model


def cache_base_model_outputs(base_model, generator, train_generator, valid_generator):
    """
    Saves base model output features for each input data sample to disc in .npy format.

    We cache base model outputs to greatly reduce training times when experimenting with network
    architectures/hyper-parameters. One of the drawbacks of this is we can't do data augmentation or fine-tune the base
    model's weights.

    :param base_model: The pre-trained base model.
    :param train_generator: Keras data generator for our train dataset.
    :param valid_generator: Keras data generator for our valid dataset.
    """
    for gen, dataset in zip([train_generator, valid_generator], ['train', 'valid']):
        print('Caching the base model\'s output features for the {} dataset to disc.'.format(dataset))

        nb_samples = len(generator.index) if dataset == 'train' else len(generator.valid_index)
        nb_batches = math.ceil(nb_samples / 32)

        for i in range(nb_batches):
            print('Caching base model\'s outputs, batch: {} of {} in the {} dataset.'.format(i, nb_batches, dataset))

            image_batch, label_batch = next(gen)
            _base_model_outputs = base_model.predict(image_batch, batch_size=len(image_batch))

            try:
                base_model_outputs = np.append(base_model_outputs, _base_model_outputs, axis=0)
                labels = np.append(labels, label_batch, axis=0)
            except NameError:
                base_model_outputs = _base_model_outputs
                labels = label_batch

        np.save(cached_base_model_outputs.format(dataset).split('.')[0], base_model_outputs)
        np.save(cached_labels.format(dataset).split('.')[0], labels)


def get_model(top_model_input_tensor, nb_classes, base_model_name, model_input_tensor=None):
    if base_model_name == 'resnet50':
        x = layers.Flatten()(top_model_input_tensor)
    elif base_model_name == 'xception':
        x = layers.GlobalAveragePooling2D()(top_model_input_tensor)
    else:
        assert False, 'Classifier network not implemented for base model: {}.'.format(base_model_name)

    x = layers.Dense(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.advanced_activations.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)

    if model_input_tensor is None:
        model_input_tensor = top_model_input_tensor

    x = layers.Dense(nb_classes, activation='softmax')(x)
    return models.Model(inputs=[model_input_tensor], outputs=[x])


@click.command()
@click.option('--valid_dir', type=str, default='dermnetnz-scraped', help='Sub-directory containing the valid data set (located in `data_dir`).')
@click.option('--cache_base_model_features', default=False, type=bool, help='Cache base model outputs for train and valid data sets to greatly reduce training times.')
@click.option('--train_top_only', default=False, type=bool, help='Train the top model (classifier network) on cached base model outputs.')
@click.option('--base_model_name', default='resnet50', type=str, help='Name of the pre-trained base model to use for transfer learning and optionally fine-tuning.')
@click.option('--fine_tune', default=False, type=bool, help='Fine tuning un-freezes top layers in the base model, training them on our data set.')
def main(valid_dir, cache_base_model_features, train_top_only, base_model_name, fine_tune):
    """
    Training.

    """
    train_dirs = set()
    labels = set()

    for dataset_dir in helpers.list_dir(data_dir, sub_dirs_only=True):
        if dataset_dir != 'data-scraped-dermnet':
            continue

        dataset_dir_path = os.path.join(data_dir, dataset_dir)
        image_details_file_path = os.path.join(dataset_dir_path, 'image_details.pickle')

        with open(image_details_file_path, 'rb') as handle:
            import pickle
            image_details = pickle.load(handle)

        for key in image_details:
            labels.add(image_details.get(key).get('parent_diagnosis'))

        train_dirs.add(os.path.join(data_dir, dataset_dir))

    labels = sorted(list(labels))
    print(labels)

    #
    # image dimensions - base models used for pre-training usually have requirements/preferences for input dimensions
    #

    img_width = 299
    img_height = 299
    img_channels = 3

    #
    # training params
    #

    batch_size = 32  # some GPUs don't have enough memory for large batch sizes
    nb_epoch = 50

    #
    # data generators
    #

    generator = image_generator.ImageGenerator(train_dirs, labels=labels, valid_split=0.2)
    train_generator = generator.image_generator(batch_size,
                                                (img_width, img_height),
                                                pre_processing_function=applications.xception.preprocess_input)
    valid_generator = generator.image_generator(batch_size,
                                                (img_width, img_height),
                                                pre_processing_function=applications.xception.preprocess_input,
                                                valid=True)

    def get_class_weights(nb_samples_per_class):
        print('Number of samples per class: {}.'.format(nb_samples_per_class))

        weights = {}
        for cls, nb_samples in nb_samples_per_class.items():
            weights[cls] = nb_samples_per_class.get(0) / nb_samples

        return weights

    # our classes are imbalanced so we need `class_weights` to scale the loss appropriately
    class_weights = get_class_weights(generator.label_sizes)
    print(class_weights)

    # load the pre-trained base model
    base_model_input_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    base_model = get_base_model(base_model_input_tensor, base_model_name)

    if cache_base_model_features:
        # feed all images (train and valid) into base model and cache its outputs
        # now during training we don't have to do forward and backward passes through the base model over and over again
        # this speeds up training dramatically and since we aren't training our base model it doesn't inhibit us much
        # Note: can't do this if we want data augmentation or want to fine-tune the base model
        cache_base_model_outputs(base_model, generator, train_generator, valid_generator)

    if train_top_only:
        # the base model outputs are fixed (cached) and serve as the input data to our top model instead of images
        model_input_tensor = None
        top_model_input_tensor = layers.Input(shape=base_model.output_shape[1:])
        assert not fine_tune, 'Fine-tuning can not be done if we are only training the top model.'
    else:
        # base model and top model will be combined into one model
        model_input_tensor = base_model_input_tensor
        top_model_input_tensor = base_model.output

        # freeze all layers in `base_model` so the pre-trained params aren't botched and prepare for fine-tuning
        for layer in base_model.layers:
            if layer.trainable_weights and fine_tune:
                try:
                    layers_to_fine_tune.append(layer)
                except NameError:
                    layers_to_fine_tune = [layer]

            layer.trainable = False

        try:
            print('Layers to fine tune: {}.'.format(layers_to_fine_tune))
        except NameError:
            pass

    # get our model (either combined or only the classifier network based on `model_input_tensor`)
    model = get_model(top_model_input_tensor, len(labels), base_model_name, model_input_tensor=model_input_tensor)

    # compile our model
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    lr = K.get_value(model.optimizer.lr)

    # print model info - this helps prevent bugs
    print(model.summary())

    # previous TensorBoard log dir should be removed before training
    try:
        shutil.rmtree(tensor_board_logs)
    except OSError:
        pass

    def on_epoch_end(epoch, _):
        """
        Anonymous function called after each epoch of training, registered with the `LambdaCallback`.

        For a batch of valid images:
            - Print confusion matrix.
            - Save image plot with predicted/true labels to disc.

        As training progresses:
            - Un-freeze layers in base model for fine-tuning.
            - TODO: Remove noisy data sets.
        """
        image_batch, label_batch = next(valid_generator)
        predicted_labels = model.predict_on_batch(image_batch)

        # categorical (one-hot encoded) => binary (class)
        label_batch = np.nonzero(label_batch)[1]
        predicted_labels = np.nonzero(predicted_labels)[1]

        # confusion matrix
        print('\n--\n{}\n--\n'.format(metrics.confusion_matrix(label_batch, predicted_labels)))

        if not train_top_only:
            # plot
            label_batch_strings = []
            for true_label, predicted_label in zip(label_batch, predicted_labels):
                label_batch_strings.append('True: {}, Predicted: {}'.format(true_label, predicted_label))

            plot_image_batch_w_labels.plot_batch(image_batch,
                                                 valid_image_batch_plot.format(epoch),
                                                 label_batch=label_batch_strings)

            # un-freeze
            if fine_tune:
                global lr
                _lr = K.get_value(model.optimizer.lr)
                if _lr < lr:
                    try:
                        layer = layers_to_fine_tune.pop()
                    except IndexError and NameError:
                        return

                    print('Un-freezing {}.'.format(layer.name))
                    layer.trainable = True

                    # model needs to be re-compiled for `layer.trainable` to take effect
                    model.compile(model.optimizer, model.loss, model.metrics)

                    lr = _lr

    def get_callbacks():
        """
        :return: A list of `keras.callbacks.Callback` instances to apply during training.

        """
        return [
            callbacks.ModelCheckpoint(model_checkpoint, monitor='val_acc', verbose=1, save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1),
            callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
            callbacks.TensorBoard(log_dir=tensor_board_logs, histogram_freq=4, write_graph=True, write_images=True)
        ]

    # train model
    if train_top_only:
        # load cached data
        base_model_outputs_train = np.load(cached_base_model_outputs.format('train'))
        labels_train = np.load(cached_labels.format('train'))
        base_model_outputs_valid = np.load(cached_base_model_outputs.format('valid'))
        labels_valid = np.load(cached_labels.format('valid'))

        model.fit(x=base_model_outputs_train,
                  y=labels_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  verbose=1,
                  callbacks=get_callbacks(),
                  validation_data=(base_model_outputs_valid, labels_valid),
                  class_weight=class_weights)

    else:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=math.ceil(len(generator.index) / batch_size),
                            epochs=nb_epoch,
                            verbose=1,
                            callbacks=get_callbacks(),
                            validation_data=valid_generator,
                            validation_steps=math.ceil(len(generator.valid_index) / batch_size),
                            class_weight=class_weights,
                            workers=4,
                            pickle_safe=True)


if __name__ == '__main__':
    main()
