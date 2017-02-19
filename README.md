# waya-dc
Image classification using deep convolutional neural networks.

For a detailed introduction to this project read [Ground-up & hands-on deep learning tutorial — diagnosing skin cancer w/ dermatologist-level accuracy](https://medium.com/@waya.ai/ground-up-hands-on-deep-learning-tutorial-diagnosing-skin-cancer-w-dermatologist-level-61a90fe9f269#.ptayqfn20).


## Getting started

#### Launching an AWS EC2 Instance

Unless you have a GPU you will most likley need to train on a cloud compute instance such as AWS EC2 (in some cases you can get away with training smaller models on a CPU). Currently we are using the [Bitfusion Ubuntu 14 TensorFlow](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0?ref=cns_srchrow) AMI, running on either a `pX.xlarge` or `gX.xlarge` instance (GPU enabled).  

After launching an EC2 instance in the [AWS EC2 Console](https://console.aws.amazon.com/console/home), `ssh` into the instance.  

```bash
$ ssh -i <AWS_KEY_INSTANCE_WAS_LAUNCHED_WITH>.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

#### Installation

```bash
$ git clone https://github.com/wayaai/waya-dc.git
$ cd waya-dc/
$ sudo pip3 install -U -r requirements.txt
$ sudo python3 setup.py develop
# Using default TensorFlow version installed on AMI, if on your PC run: $ sudo pip3 install -U tensorflow
```

#### Getting data sets

```bash
$ python3 wayadc/utils/get_datasets.py
```

#### Training model

```bash
$ python3 train.py --help  # To see available options
$ python3 train.py
```

#### Notes

* Make sure you use Waya.ai's [Keras fork](https://github.com/wayaai/keras) (see [requirements.txt](https://github.com/wayaai/waya-dc/blob/master/requirements.txt)).
* [Getting started w/ tmux on AWS](https://medium.com/towards-data-science/deep-learning-aws-ec2-tmux-3b96777016e2#.uogw5eavz) will help you when you really start training models.
* You may prefer [Running a Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) on the AWS instance.

#### About Waya.ai
[Waya.ai](http://waya.ai) is a company whose vision is a world where medical conditions are addressed early on, in their infancy. This approach will shift the health-care industry from a constant fire-fight against symptoms to a preventative approach where root causes are addressed and fixed. Our first step to realize this vision is easy, accurate and available diagnosis. Our current focus is concussion diagnosis, recovery tracking & brain health monitoring. Please get in contact with me if this resonates with you!
