import os
import tensorflow as tf
import matplotlib
from practice4.object_recognize_util import ObjectDetection

matplotlib.use('Agg')
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

MODELS = [
    'ssd_mobilenet_v1_coco_2018_01_28',
    'ssd_inception_v2_coco_2018_01_28',
    'mask_rcnn_resnet101_atrous_coco_2018_01_28',
    'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
    'faster_rcnn_resnet50_lowproposals_coco_2018_01_28'
]
PATH_TO_TEST_IMAGES_DIR = '/image'
TEST_IMAGE_PATHS = [os.path.join(os.getcwd() + PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1, 9)]
factory = ObjectDetection('predict', os.path.join(os.getcwd(), os.pardir), (40, 40))
factory.recognize_by_model(MODELS, TEST_IMAGE_PATHS)

