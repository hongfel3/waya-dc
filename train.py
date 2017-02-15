"""
Image classification using deep convolutional neural networks.

Note: Only Python 3 support currently.
"""

import math
import os

import click
from keras import applications
from keras import callbacks
from keras import layers
from keras import models
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

from dlutils import plot_image_batch_w_labels
from wayadc.utils import helpers


#
# directories and paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')  # cached base model outputs, model checkpoints, etc... saved in this dir
cached_base_model_outputs = os.path.join(cache_dir, 'base_model_outputs_{}.tfrecords')
model_checkpoint = os.path.join(cache_dir, 'model_checkpoint.h5')  # models are saved during training as they improve


def get_base_model(input_tensor, base_model_name):
    """
    Get a pre-trained base model for transfer learning (and optionally fine tuning).

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


def cache_base_model_outputs(base_model, train_generator, valid_generator):
    """
    Saves base model output features for each input data sample to disc in TFRecord file format.

    Each record w/in the TFRecord file is a serialized Example proto.
    See: https://www.tensorflow.org/how_tos/reading_data/ for more info.

    We cache base model outputs to greatly reduce training times when experimenting with
    network architectures/hyper-parameters.

    Note: One of the drawbacks of this is we can't do data augmentation or fine-tune the base model's weights.
    See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#protobuf-library-related-issues

    :param base_model: The pre-trained base model.
    :param train_generator: Keras data generator for our train dataset.
    :param valid_generator: Keras data generator for our valid dataset.
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    for generator, dataset in zip([train_generator, valid_generator], ['train', 'valid']):
        print('Saving base model\'s output features for the {} dataset to disc as a TFRecords file.'.format(dataset))
        writer = tf.python_io.TFRecordWriter(cached_base_model_outputs.format(dataset))

        nb_batches = math.ceil(generator.nb_sample / generator.batch_size)
        for i in range(nb_batches):
            print('Caching base model\'s outputs, batch: {} of {} in the {} dataset.'.format(i, nb_batches, dataset))

            image_batch, label_batch = generator.next()
            base_model_outputs = base_model.predict(image_batch, batch_size=len(image_batch))

            for j, base_model_output in enumerate(base_model_outputs):
                base_model_output_raw = base_model_output.tostring()
                label_raw = label_batch[j].tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'base_model_output_features': _bytes_feature(base_model_output_raw),
                    'label': _bytes_feature(label_raw)}))

                writer.write(example.SerializeToString())

        writer.close()


def get_model(model_input_tensor, nb_classes, base_model_name):
    if base_model_name == 'resnet50':
        x = layers.Flatten()(model_input_tensor)
    elif base_model_name == 'xception':
        x = layers.GlobalAveragePooling2D()(model_input_tensor)
    else:
        assert False, 'Classifier network not implemented for base model: {}.'.format(base_model_name)

    x = layers.Dense(nb_classes, activation='softmax')(x)
    return models.Model(input=model_input_tensor, output=x)


@click.command()
@click.option('--train_dir', type=str, help='Full path of the directory containing the train data set.')
@click.option('--valid_dir', type=str, help='Full path of the directory containing the valid data set.')
@click.option('--cache_base_model_features', default=False, type=bool,
              help='Cache base model outputs for train and valid data sets to greatly reduce training times.')
@click.option('--train_top_only', default=False, type=bool,
              help='Train the top model (classifier network) on cached base model outputs.')
@click.option('--base_model_name', default='xception', type=str,
              help='Name of the pre-trained base model to use for transfer learning and optionally fine-tuning.')
@click.option('--fine_tune', default=False, type=bool,
              help='Fine tuning un-freezes top layers in the base model, training them on our data set.')
def main(train_dir, valid_dir, cache_base_model_features, train_top_only, base_model_name, fine_tune):
    #
    # image dimensions
    #

    img_width = 224
    img_height = 224
    img_channels = 3

    #
    # training params
    #

    batch_size = 32
    nb_epoch = 50

    #
    # data generators
    #

    # there are lots of optional params for data augmentation that can be used here
    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,  # normalize data so entries are in range [-1, 1]
        dim_ordering='tf')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': 'categorical',
                                  'batch_size': batch_size}

    train_generator = data_generator.flow_from_directory(
        directory=train_dir,
        **flow_from_directory_params)

    valid_generator = data_generator.flow_from_directory(
        directory=valid_dir,
        **flow_from_directory_params)

    nb_classes = train_generator.nb_class
    assert train_generator.nb_class == valid_generator.nb_class, \
        'Train and valid data sets must have the same number of classes.'

    nb_train_samples = train_generator.nb_sample
    nb_valid_samples = valid_generator.nb_sample

    # load the pre-trained base model and freeze it so its weights aren't messed up during training
    base_model_input_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    base_model = get_base_model(base_model_input_tensor, base_model_name)
    base_model.trainable = False

    if cache_base_model_features:
        # feed all images (train and valid) into base model and cache its outputs
        # now during training we don't have to do forward and backward passes through the base model over and over again
        # this speeds up training dramatically and since we aren't training our base model it doesn't inhibit us much
        # Note: can't do this if we want data augmentation or want to fine-tune the base model
        cache_base_model_outputs(base_model, train_generator, valid_generator)

    if train_top_only:
        # the base model outputs are fixed (cached) and serve as the input data to our top model instead of images
        model_input_tensor = layers.Input(shape=base_model.output_shape[1:])
    else:
        # base model and top model will be combined into one model. the inputs to this model are images
        model_input_tensor = base_model_input_tensor

    if fine_tune:
        # make sure top model is trained before starting to fine tune the base model so large gradients don't botch the
        # base models pre-trained weights
        # the learning rate should also be decreased before fine tuning
        # TODO: make a keras contribution allowing making base model layers trainable upon loss plateaus
        assert False, 'Fine tuning not implemented yet.'

    # get our model (either combined or only the classifier ntwk based on `model_input_tensor`) and compile for training
    model = get_model(model_input_tensor, nb_classes, base_model_name)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    # Note: this is very slow if not using a GPU and `train_top_only` should not be used in this case
    if train_top_only:
        def base_model_output_generator(dataset):
            """
            Generator function that yields batches of base model output features and corresponding labels.

            :param dataset: Whether to yield batches of data/labels from the 'train' or 'valid' TFRecordFile.
            """
            # infinite, shuffled iteration over data stored in the TFRecordFile
            for serialized_example in tf.python_io.tf_record_iterator(cached_base_model_outputs.format(dataset)):
                base_model_output_features_batch = []
                label_batch = []

                # must register a tensorflow session to convert tensors read from TFRecordFiles to numpy arrays
                with tf.Session() as _:
                    while len(base_model_output_features_batch) != batch_size:
                        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py#L50
                        features = tf.parse_single_example(
                            serialized_example,
                            features={'base_model_output_features': tf.FixedLenFeature([], tf.string),
                                      'label': tf.FixedLenFeature([], tf.string)}
                        )

                        b = tf.decode_raw(features.get('base_model_output_features'), out_type=tf.float32)
                        b = tf.reshape(b, shape=base_model.output_shape[1:])

                        l = tf.decode_raw(features.get('label'), out_type=tf.float32)
                        l = tf.reshape(l, shape=(nb_classes, ))

                        base_model_output_features_batch.append(b.eval())
                        label_batch.append(l.eval())

                yield np.asarray(base_model_output_features_batch), np.asarray(label_batch)

        # since we are only training the top model, over-ride image generators with cached base model output generators
        train_generator = base_model_output_generator('train')
        valid_generator = base_model_output_generator('valid')

    class PlotModelPredictions(callbacks.Callback):
        """
        Plot a batch of validation images along w/ their true label and model's prediction after each epoch of training.

        """
        def on_epoch_end(self, epoch, logs={}):
            image_batch, label_batch = valid_generator.next()
            label_batch = np_utils.categorical_probas_to_classes(label_batch)

            predicted_labels = self.model.predict_on_batch(image_batch)
            predicted_labels = np_utils.categorical_probas_to_classes(predicted_labels)

            label_batch_strings = []
            for true_label, predicted_label in zip(label_batch, predicted_labels):
                label_batch_strings.append('True: {}, Predicted: {}'.format(true_label, predicted_label))

            plot_image_batch_w_labels.plot_batch(image_batch, os.path.join(cache_dir, 'plot_{}.png'.format(epoch)),
                                                 label_batch=label_batch_strings)

    def get_callbacks():
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=1),
            callbacks.ModelCheckpoint(model_checkpoint, monitor='val_acc', save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1),
            PlotModelPredictions()
        ]

    # our classes are imbalanced so we need `class_weights` to scale the loss appropriately
    classes = helpers.list_dir(train_dir, sub_dirs_only=True)
    print(classes)
    class_sizes = helpers.get_class_sizes(train_dir, valid_dir, classes)
    print(class_sizes)
    class_weights = helpers.get_class_weights(class_sizes, classes)
    print(class_weights)

    # train our model
    model.fit_generator(train_generator, nb_train_samples, nb_epoch=nb_epoch, verbose=1, callbacks=get_callbacks(),
                        validation_data=valid_generator, nb_val_samples=nb_valid_samples, class_weight=class_weights)


if __name__ == '__main__':
    main()
