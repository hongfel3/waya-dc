"""
TODO: Fix these tests for users who don't have access to my private s3 buckets.
TODO: Use pytest framework more effectively.

To run: $ pytest tests/
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


def test_cleanup():
    shutil.rmtree(TEST_DATA_DIR)
    assert not os.path.isdir(TEST_DATA_DIR)
