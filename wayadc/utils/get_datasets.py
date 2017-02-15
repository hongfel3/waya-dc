import os

from wayadc.utils import helpers


path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(path, '../../data')  # data sets are saved in this dir


def main():
    aws_bucket = 'waya-sig-ai'

    helpers.download_aws_s3_bucket(aws_bucket, data_dir)
    helpers.unzip_files_in_dir(data_dir)


if __name__ == '__main__':
    main()
