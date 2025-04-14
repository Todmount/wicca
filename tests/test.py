import os

import cv2
import pandas as pd
import tensorflow.keras.applications as apps

from wicca.config.constants import *
import wicca.result_manager as rmgr
from wicca.data_loader import load_models
from wicca.wavelet_coder import HaarCoder
from wicca.classifying_tools import ClassifierProcessor

# Settings
pd.set_option('display.float_format', '{:.5f}'.format) # Nice format for dataframes output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # WARNING. Disabling gpu

standard_shape_models_dict = {
    'MobileNetV2': apps.mobilenet_v2.MobileNetV2,
    # 'VGG16': apps.vgg16.VGG16,
    # 'VGG19': apps.vgg19.VGG19,
    'ResNet50': apps.resnet50.ResNet50,
    # 'ResNet101': apps.resnet.ResNet101,
    # 'DenseNet169': apps.densenet.DenseNet169,
    # 'DenseNet201': apps.densenet.DenseNet201,
    # 'NASNetMobile': apps.nasnet.NASNetMobile,
    'ResNet152': apps.resnet.ResNet152,
    # 'EfficientNetB0': apps.efficientnet.EfficientNetB0,
    # 'DenseNet121': apps.densenet.DenseNet121,
    # 'EfficientNetB1': (apps.efficientnet.EfficientNetB1, {'shape': (240, 240)})
}

single_model = {
    "MobileNetV2": apps.mobilenet_v2.MobileNetV2
}

# classifiers = load_models(models_dict)
classifiers = load_models(standard_shape_models_dict)

# Depths
depth = 5
depth_tuple = (3, 4)
depth_range = [5,6]
depth_list = [2, 6]
depth_tuple_list = [(3, 4), (4, 5), (5, 6)] # would fail

data_folder = PROJECT_ROOT / 'data' / '10test'  # for test purposes
res_folder = PROJECT_ROOT / 'results' / 'test'

res_folder_nonexist = f'{res_folder}_nonexist'

if __name__ == '__main__':
    processor = ClassifierProcessor(
        data_folder=data_folder,  # Directory containing images to process
        wavelet_coder=HaarCoder(),  # Defines our wavelet
        transform_depth=depth_range,  # Defines the depth of transforming
        top_classes=5,  # Defines top classes for comparison
        interpolation=cv2.INTER_AREA,  # Interpolation method. See https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
        result_manager=rmgr,  # Result manager module
        results_folder=res_folder,  # Where to save results
        log_info=True,  # Output related info. Enabled by default
        parallel=5, # What means infinity
        batch_size=30 # Size of image batch for classifier
    )
    processor.process_classifiers(classifiers, timeout=3600)