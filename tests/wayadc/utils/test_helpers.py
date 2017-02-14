"""
TODO: Fix these tests for users who don't have access to my private s3 buckets.

"""

import os
import shutil

from wayadc.utils import helpers


PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(PATH, '../test_data')
TEST_S3_DIR = os.path.join(TEST_DATA_DIR, 's3')


def test_download_aws_s3_object():
    # Test 1: AWS object and directory it is to be stored in do not exist.
    helpers.download_aws_s3_object('doctor-ai-moles', 'manually_collected.zip', TEST_S3_DIR)
    assert os.path.isfile(os.path.join(TEST_S3_DIR, 'manually_collected.zip'))

    # Test 2: AWS object already exists.
    helpers.download_aws_s3_object('doctor-ai-moles', 'manually_collected.zip', TEST_S3_DIR)
    assert os.path.isfile(os.path.join(TEST_S3_DIR, 'manually_collected.zip')) and \
        len(helpers.list_dir(TEST_S3_DIR)) == 1

    # Test 3: AWS object already exists, but force the download (update it).
    helpers.download_aws_s3_object('doctor-ai-moles', 'manually_collected.zip', TEST_S3_DIR, force=True)
    assert os.path.isfile(os.path.join(TEST_S3_DIR, 'manually_collected.zip')) and \
        len(helpers.list_dir(TEST_S3_DIR)) == 1


def test_unzip_files_in_dir():
    assert os.path.isfile(os.path.join(TEST_S3_DIR, 'manually_collected.zip'))

    # Test 1: A single zip file exists in directory.
    helpers.unzip_files_in_dir(TEST_S3_DIR)
    assert os.path.isdir(os.path.join(TEST_S3_DIR, 'manually_collected'))

    # Test 2: The zip file, and it's uncompressed directory exist in directory.
    helpers.unzip_files_in_dir(TEST_S3_DIR)
    assert os.path.isdir(os.path.join(TEST_S3_DIR, 'manually_collected')) and \
        len(helpers.list_dir(TEST_S3_DIR, sub_dirs_only=True)) == 1


def test_list_dir():
    test_dir = os.path.join(TEST_S3_DIR, 'manually_collected')
    expected_sub_dirs = ['atypical', 'blue', 'carcinoma', 'common', 'melanoma', 'spitz']

    assert os.path.isdir(test_dir)

    # Test 1
    assert helpers.list_dir(test_dir) == expected_sub_dirs

    # Test 2
    assert helpers.list_dir(test_dir, sub_dirs_only=True) == expected_sub_dirs

    # Test 3
    assert helpers.list_dir(test_dir, images_only=True) == []

    # Test 4
    try:
        helpers.list_dir(test_dir, sub_dirs_only=True, images_only=True)
        assert False
    except AssertionError as _:
        pass
    except Exception as e:
        assert False, e


def test_copy_image():
    src_dir = os.path.join(os.path.join(TEST_S3_DIR, 'manually_collected'), 'atypical')

    images = helpers.list_dir(src_dir, images_only=True)

    # Test 1
    helpers.copy_image(images[0], src_dir, TEST_DATA_DIR)
    assert os.path.isfile(os.path.join(TEST_DATA_DIR, images[0]))
    assert os.path.isfile(os.path.join(src_dir, images[0]))
    assert len(helpers.list_dir(TEST_DATA_DIR, images_only=True)) == 1

    # Test 2: Image to copy already exists in dest_dir.
    helpers.copy_image(images[0], src_dir, TEST_DATA_DIR)
    assert os.path.isfile(os.path.join(TEST_DATA_DIR, images[0]))
    assert len(helpers.list_dir(TEST_DATA_DIR, images_only=True)) == 1


def test_get_file_path_with_extension():
    dir = os.path.join(os.path.join(TEST_S3_DIR, 'manually_collected'), 'atypical')

    images = helpers.list_dir(dir, images_only=True)
    test_image = images[0]
    test_image_no_ext = test_image.split('.')[0]

    assert helpers.get_file_path_with_extension(os.path.join(dir, test_image_no_ext)) == os.path.join(dir, test_image)


def test_get_class_weights():
    classes = ['c1', 'c2']

    # Test 1.
    class_sizes = {'train': {classes[0]: 100, classes[1]: 100}, 'valid': {}}
    class_weights = helpers.get_class_weights(class_sizes, classes)

    assert class_weights == {0: 1.0, 1: 1.0}

    # Test 2.
    class_sizes = {'train': {classes[0]: 200, classes[1]: 50}, 'valid': {}}
    class_weights = helpers.get_class_weights(class_sizes, classes)

    assert class_weights == {0: 1.0, 1: 4.0}

    # Test 3.
    class_sizes = {'train': {classes[0]: 50, classes[1]: 200}, 'valid': {}}
    class_weights = helpers.get_class_weights(class_sizes, classes)

    assert class_weights == {0: 1.0, 1: 0.25}

    # Test 4.
    classes.append('c3')
    class_sizes = {'train': {classes[0]: 200, classes[1]: 200, classes[2]: 200},
                   'valid': {classes[0]: 12, classes[1]: 34}}
    class_weights = helpers.get_class_weights(class_sizes, classes)

    assert class_weights == {0: 1.0, 1: 1.0, 2: 1.0}


# TODO: Figure out pytest framework...
def test_cleanup():
    shutil.rmtree(TEST_DATA_DIR)
    assert not os.path.isdir(TEST_DATA_DIR)
