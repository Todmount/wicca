import os
import cv2

from pathlib import Path
import tensorflow.keras.applications as apps

import utility.result_manager as rsltmgr
import utility.visualization as viz
from utility.data_loader import load_image
from utility.wavelet_coder import HaarCoder
from utility.classifying_tools import load_models, ClassifierProcessor
from settings.constants import *

import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

if __name__ == '__main__':
    models_to_load = {
        # Mobile Networks (standard 224×224)
        'MobileNetV2': apps.mobilenet_v2.MobileNetV2,

        # VGG family (standard 224×224)
        'VGG16': apps.vgg16.VGG16,
        'VGG19': apps.vgg19.VGG19,

        # ResNet family (standard 224×224)
        'ResNet50': apps.resnet50.ResNet50,
        'ResNet101': apps.resnet.ResNet101,
        'ResNet152': apps.resnet.ResNet152,

        # DenseNet family (standard 224×224)
        'DenseNet121': apps.densenet.DenseNet121,
        'DenseNet169': apps.densenet.DenseNet169,
        'DenseNet201': apps.densenet.DenseNet201,

        # NasNet family (custom shapes)
        'NASNetMobile': apps.nasnet.NASNetMobile,  # Uses 224×224
        'NASNetLarge': (apps.nasnet.NASNetLarge, {'shape': (331, 331)}),

        # Inception family (custom shapes)
        'InceptionV3': (apps.inception_v3.InceptionV3, {'shape': (299, 299)}),
        'InceptionResNetV2': (apps.inception_resnet_v2.InceptionResNetV2, {'shape': (299, 299)}),

        # Xception (custom shape)
        'Xception': (apps.xception.Xception, {'shape': (299, 299)}),
    }

    models4test = {
        # VGG family (standard 224×224)
        # 'VGG16': apps.vgg16.VGG16,
        'VGG19': apps.vgg19.VGG19,
        'MobileNetV2': apps.mobilenet_v2.MobileNetV2,
        'ResNet152': apps.resnet.ResNet152,
        # 'InceptionResNetV2': (apps.inception_resnet_v2.InceptionResNetV2, {'shape': (299, 299)}),
        # 'NASNetLarge': (apps.nasnet.NASNetLarge, {'shape': (331, 331)}),
        'DenseNet121': apps.densenet.DenseNet121,
    }

    # classifiers = load_models(models_to_load) # around 1 minute
    classifiers = load_models(models4test)

    # path =  PROJECT_ROOT / 'data' / 'orig'
    path = PROJECT_ROOT / 'data' / '4test'  # for test purposes
    res_folder = PROJECT_ROOT / 'results'

    depths = range(3, 6)
    processor = ClassifierProcessor(path=path,
                                    coder=HaarCoder(),  # defines our wavelet
                                    depth=depths,  # defines the depth of transforming
                                    top=5,  # defines top classes for comparison
                                    interpolation=cv2.INTER_AREA,  # ATTENTION
                                    results_folder=res_folder,
                                    rsltmgr=rsltmgr)

    ## must work for all classifiers
    # result_all_depths = processor.process_classifiers(classifiers, timeout=100) # results for multiple all

    ## must work for single
    # result_VGG19 = processor.process_single_classifier("VGG19", classifiers["VGG19"], timeout=3601)

    ## must work flawlessly
    # results = processor.process_classifiers(classifiers["VGG19"])

    ## must work, not typical
    # result_VGG16 = processor.process_classifiers({"VGG16": classifiers["VGG16"]},timeout = 3601)

    ## must not work
    # result_MobileNetV2 = processor.process_classifiers(classifiers["MobileNetV2"])

    ## must not work
    # result_VGG19 = processor.process_single_classifier(classifiers["VGG19"])

    """VISUALIZATION"""
    # load image
    list_dir = [f.name for f in path.iterdir()]
    print(f'Number of images: {len(list_dir)} \n')
    name = list_dir[0]
    sample = load_image(path / name)

    # viz.show_image_vs_icon(sample,depths, coder=HaarCoder())

    # viz.dwt_visualization(sample, depths, border_width=2, border_color=(0, 0, 255), coder=HaarCoder())

    # x = rsltmgr.compare_summaries(res_folder, classifiers, depths)

    # comparison = rsltmgr.compare_summaries(res_folder, classifiers, 5, 'mean')
    # names, similar_classes_pct = rsltmgr.extract_from_comparison(comparison, 'similar classes (%)')
    # names, similar_best_class = rsltmgr.extract_from_comparison(comparison, 'similar best class')

    # viz.plot_metric_radar(names, similar_classes_pct, 'Similar classes (%)')
    # viz.plot_metric_radar(names, similar_best_class, 'Similar best class')

    # viz.plot_compare_metrics(names, similar_classes_pct, similar_best_class)

    # viz.visualize_comparison(x, SIM_CLASSES_PERC, title="Similar classes, % (mean)")
    # image = Path(path / name)

    # Looking for a way to compare size of source and compressed items
    # print(image.stat().st_size)

    res_folder_nonexist = Path(res_folder / '3')
    yu = rsltmgr.load_summary_results(res_folder, 'MobileNetV2', 5)
    print(yu)
