import os
import zipfile

import boto3


# Should always correspond with https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L630.
image_ext_whitelist = ['bmp', 'jpeg', 'jpg', 'png']


def list_dir(dir_path, sub_dirs_only=False, images_only=False):
    assert not (sub_dirs_only and images_only)

    def file_name_filter(item):
        if item.startswith('.') or item.startswith('__'):
            return False
        if sub_dirs_only and not os.path.isdir(os.path.join(dir_path, item)):
            return False
        elif images_only and item.split('.')[-1].lower() not in image_ext_whitelist:
            return False

        return True

    return sorted(filter(file_name_filter, os.listdir(dir_path)))


def download_aws_s3_object(aws_bucket, object_name, dest_dir, force=False):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(aws_bucket)
    obj = bucket.Object(object_name)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    object_file_path = os.path.join(dest_dir, object_name)
    if not os.path.isfile(object_file_path) or force:
        print('Downloading: {} to: {}.'.format((aws_bucket, object_name), dest_dir))
        obj.download_file(object_file_path)


def download_aws_s3_bucket(aws_bucket, dest_dir, force=False):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(aws_bucket)
    for object_name in bucket.objects.all():
        download_aws_s3_object(aws_bucket, object_name.key, dest_dir, force=force)


def unzip_files_in_dir(dir_path):
    def unzip_file(file):
        if file.endswith('.zip'):
            print('Unzipping: {}.'.format(file))
            with zipfile.ZipFile(os.path.join(dir_path, file)) as zf:
                zf.extractall(dir_path)

    list(map(unzip_file, list_dir(dir_path)))