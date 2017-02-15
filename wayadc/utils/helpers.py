import glob
import os
import shutil
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


def copy_image(image_name, src_dir, dest_dir, move=False):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    image_path = os.path.join(src_dir, image_name)
    if not os.path.isfile(image_path):
        image_path = get_file_path_with_extension(image_path)
        if not image_path:
            return

    if move:
        shutil.move(image_path, dest_dir)
    else:
        shutil.copy(image_path, dest_dir)


def get_file_path_with_extension(file_path_without_extension):
    list_of_image_paths = glob.glob('{}.*'.format(file_path_without_extension))
    if len(list_of_image_paths) != 1:
        print('{}, {}'.format(file_path_without_extension, list_of_image_paths))
        return

    return list_of_image_paths[0]


def number_images_in_dir_tree(root):
    count = 0

    for path, sub_dirs, files in os.walk(root):
        for file in files:
            if file.split('.')[-1].lower() in image_ext_whitelist:
                count += 1

    return count


def get_class_sizes(train_dir, valid_dir, classes):
    sizes = {'train': {}, 'valid': {}}

    for cls in classes:
        sizes.get('train')[cls] = len(list_dir(os.path.join(train_dir, cls), images_only=True))
        sizes.get('valid')[cls] = len(list_dir(os.path.join(valid_dir, cls), images_only=True))

    return sizes


def get_class_weights(class_sizes, classes):
    class_weights = {}

    for i, cls in enumerate(classes):
        class_weights[i] = class_sizes.get('train').get(classes[0]) / class_sizes.get('train').get(cls)

    return class_weights
