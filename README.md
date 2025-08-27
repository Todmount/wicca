<!-- Project name: caps "WICCA" and "Wavelet-based Image Compression & Classification Analysis" underneath both in 41b883 color-->
<p align="center" title="Project name" alt="WICCA: Wavelet-based Image Compression and Classification Analysis">
   <img src="https://res.cloudinary.com/dxteec1w4/image/upload/v1756250680/wicca_ohxi8t.png" >
</p>

<!-- Project specific badges -->
<p align="center">
  <a href="https://python.org" title="Supported python versions" alt="Supported python versions">
    <img src="https://img.shields.io/badge/Python-3.12+-blue.svg">
  </a>
  <a href="LICENSE" title="License" alt="License">
    <img src="https://img.shields.io/badge/License-GNU%20GPL-yellow">
  </a>
  <a href="https://github.com/psf/black" title="Code style" alt="Code style: black">
    <img src="https://img.shields.io/badge/Code%20Style-black-000000.svg">
  </a>
</p>

## Overview
WICCA is a research-driven framework for investigating how wavelet-based image compression and scaling affect the classification performance of pretrained models. WICCA leverages [Discrete Wavelet Transform (DWT)](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) to generate compressed icons — small representations of images extracted from wavelet decomposition. At the moment, [Haar wavelet](https://en.wikipedia.org/wiki/Haar_wavelet) compression is implemented as the initial method, with extensions to other wavelets planned. compression method

## Project Goals
### Global Goal
- Systematically evaluate how different wavelet compression techniques influence classification across a variety of pretrained models

### Current Goals
- Analyze how Haar wavelet compression affects classification performance
- Compare classification results between original high-resolution images and their compressed counterparts

## How It Works
- Dataset preparation: Large-scale, high-resolution images (≥2K) are sourced
- Wavelet compression: Haar decomposition is applied to extract representative icons
- Model inference: Pre-trained CNN classifiers are used to evaluate both original and compressed images
- Prediction analysis: Performance is compared using top-1 accuracy match, top-5 class intersection, and prediction similarity
- Results storage: Outcomes are structured in Pandas DataFrames and exported as .csv

## Core Functionality
✅ Supports large-image datasets  
✅ Supports multiple CNN architectures  
✅ Uses pre-trained models: no model training required  
✅ Enables structured analysis, comparing classification results between original images and their wavelet-compressed counterparts  
✅ Developed to use in Jupyter Notebook, but can be used as a CLI

## Installation

<details>
   <summary>Requirements</summary>

   - Python 3.12+
   - Conda

</details>

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/WICCA.git
   cd WICCA
   ```

2. **Set up a conda environment**
   ```bash
   conda create -n wicca python=3.12
   conda activate wicca
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

**or**   

Download Docker image (will be provided later)

## Usage
> Please refer to the [demonstration notebook](https://github.com/Todmount/wicca/blob/main/demo.ipynb) to see all available functionality  

Basic usage is as follows:   

0. <details>
      <summary>Imports</summary>
   
   ```python3
   import os

   import cv2
   import pandas as pd
   import matplotlib.pyplot as plt
   import tensorflow.keras.applications as apps
   from pathlib import Path
   
   import wicca.visualization as viz
   import wicca.result_manager as rmgr
   from wicca.data_loader import load_image, load_models
   from wicca.wavelet_coder import HaarCoder
   from wicca.classifying_tools import ClassifierProcessor
   from wicca.config.constants import SIM_CLASSES_PERC, SIM_BEST_CLASS, RESULTS_FOLDER
   ```
   </details>
   

2. Define the models dictionary in the format as in the example, and load it
   
   ```python3
   models_dict = {
      # Mobile Networks (standard 224×224)
      'MobileNetV2': apps.mobilenet_v2.MobileNetV2,
      # NasNet family (custom shapes)
       'NASNetMobile': apps.nasnet.NASNetMobile,  # Uses 224×224
       'NASNetLarge': (apps.nasnet.NASNetLarge, {'shape': (331, 331)}),
   }
   classifiers = load_models(models_dict)
   ```
3. Define the processor and call it
   
   ```python3
   processor = ClassifierProcessor(
       data_folder='data/originals',  # Set directory containing images to process
       wavelet_coder=HaarCoder(),  # Set our wavelet
       transform_depth=range(1,6),  # Set the depth of transforming; accepts int | range | list
       top_classes=5,  # Set top classes for comparison
       interpolation=cv2.INTER_AREA,  # Set interpolation method
       results_folder='results',  # Set results folder
       log_info=True,  # Print input-related info; enabled by default
       batch_size=30 # Set size of image batch for classifier
   )
   processor.process_classifiers(classifiers, timeout=3600) # timeout for whole process, in seconds
   ```
   <details>
      <summary>Output: </summary>
      
      ```bash
         **Image Processing Configuration**
         Note: For image statistics, a sample of 50 random images was taken.
         You may change the sample size [MAX_INFO_SAMPLE_SIZE] in the config.constants module.
         
         Data folder: data\originals
         Number of images: 130
         Mean image dimensions: 8284x6393 px
         Mean image resolution: 52.7 MP (52685873 pixels)
         Transform depth: (2, 3, 4, 5, 6)
         Interpolation: cv2.INTER_AREA
         Top classes: 5
         Results folder: C:\MyProjects\WICCA\results
      
         Processing depth 2:   100%|▮▮▮▮▮▮▮▮▮▮|[15:00]   
         Processing depth 3:   100%|▮▮▮▮▮▮▮▮▮▮|[17:00]   
         Processing depth 4:   100%|▮▮▮▮▮▮▮▮▮▮|[20:00]   
         Processing depth 5:   100%|▮▮▮▮▮▮▮▮▮▮|[23:00]   
         Processing depth 6:   100%|▮▮▮▮▮▮▮▮▮▮|[25:00]
      
         Total processing time: 1 hour 30 minutes
      ```
      >Runtime depends heavily on dataset size, transform depth, and available hardware.
   
   </details>
   
5. See the results in table format
   ```python3
   results['MobileNetV2'] # direct call to DataFrame will output results for all depth
   rmgr.load_summary_results(results_folder='results', depth = 5, classifier_name='EfficientNetB0') # load csv with specific depth
   ```
   <details>
      <summary>Output: </summary>

   ||stat|similar classes (count)|similar classes (%)|similar best class|
   |-|-|-|-|-|
   |0|mean|4.24|84.92|83.84|
   |2|min|1.00|20.00|0.00|
   |3|max|5.00|100.00|100.00|

   > The output summarizes mean, min, and max similarity metrics across images

   </details>
   
6. You could visualize results utilizing built-in functionality

   ```python3
   comparison = rmgr.compare_summaries(res_default, classifiers, 5, 'mean')
   names, similar_classes_pct = rmgr.extract_from_comparison(comparison, 'similar classes (%)')
   
   # similar classes
   viz.plot_metric_radar(
    names=names, 
    metric=similar_classes_pct,
    title='Best 5 Classes Similarity',
    min_value=75, max_value=95
   )
   ```
   
   <details>
      <summary> Output:</summary>
      <img src="https://cdn.imgpile.com/f/VJXdkqn_xl.png" alt="output_example.png">
   </details>

   ```python3
   comparison_data = rmgr.compare_summaries(res_default, classifiers, depth_range, "mean")
   viz.visualize_comparison(
       comparison_data=comparison_data,
       metric=SIM_CLASSES_PERC,
       title="Best 5 Classes Similarity Heatmap"
   )
   ```
   <details>
      <summary>Output:</summary>
      <img src="https://cdn.imgpile.com/f/QrO6Bhc_xl.png" alt="output_heatmap.png">
   </details>
   

## Roadmap
- [x] Implement Haar wavelet compression
- [x] Implement comparison functionality for various conversion depths and classifiers
- [x] Add a side-by-side visualization of the original image and its icon
- [ ] Dockerize the project
- [ ] Write a detailed documentation
- [ ] Extend to other wavelets (Daubechies, Coiflet, etc.)
- [ ] Optimize for large-scale datasets

<!-- Results -->
<details>
   <summary><h2>Results & Insights</h2></summary>
   
   - Haar wavelet compression yields icons that preserve structural features, enabling effective classification
   - While image size is significantly reduced, essential visual information remains intact  
   - Compression reduces computational cost but may slightly impact accuracy, depending on the classifier  
   - Classification accuracy shows model-dependent sensitivity to compression; architectures such as ResNet and NASNet maintain robustness  
   
</details>

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request

## Contact
You could find relevant contact info on my [GitHub profile](https://github.com/Todmount) 
