# WICCA : Wavelet-based Image Compression & Classification Analysis

## 📌 Overview
WICCA is a research-driven framework that investigates how wavelet-based image compression affects classification performance. WICCA leverages [Discrete Wavelet Transform (DWT)](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) to generate compressed icons—small representations of images extracted from wavelet decomposition. Currently, the framework is centered on [Haar wavelet](https://en.wikipedia.org/wiki/Haar_wavelet) compression

## 🎯 Project Goals
### 🌍 Global Goal
- Evaluate the impact of various wavelet compression techniques on different classifiers

### 🔍 Current Goal
- Analyze how Haar wavelet compression influences classification performance
- Compare classification results between original high-resolution images and their compressed counterparts

## 🛠 How It Works
1. **Dataset Preparation**: Large images (>2K resolution) are sourced from Pixabay
2. **Wavelet Compression**: The Haar wavelet (depth = 5) is applied, extracting the icon from the multi-resolution decomposition
3. **Model Inference**: Pre-trained CNN classifiers are used to classify both the original and icon images
4. **Prediction Analysis**: Classification results are compared using three key metrics:
   - Top-1 accuracy match (same best class for both images).
   - Top-5 class intersection (overlap in the best 5 predictions).
   - Percentage of similar predicted classes.
5. **Results Storage**: All results are stored in a Pandas DataFrame for further analysis.

## 🚀 Features
✅ Supports large image datasets  
✅ Uses pre-trained models – No model training required  
✅ Supports multiple CNN architectures  
✅ Implements wavelet-based image compression    
✅ Enables structured analysis, comparing classification results between original images and their wavelet-compressed counterparts  

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/WICCA.git
cd WICCA

# Install dependencies
pip install -r requirements.txt
```

**or**   

Download docker image (will be provided later)

## 🏗 Usage
As for now, please refer to the **demo.ipynb**

## 🔄 Roadmap
- [x] Implement Haar wavelet compression
- [x] Implement comparison functionality by various conversion depth and classifiers
- [x] Add a side-by-side visualization of the original image and its icon
- [ ] Implement minimal Flask MVP
- [ ] Dockerize the project
- [ ] Extend to other wavelets (Daubechies, Coiflet, etc.)
- [ ] Optimize for large-scale datasets

## 📈 Results & Insights
### ✅ Key Findings (so far)
- Haar wavelet compression produces an image icon that retains structural features, making classification feasible  
- The transformation significantly reduces image size but retains recognizable features  
- Compression reduces computational cost but may slightly impact accuracy, depending on the classifier  
- Some models (ResNet, NASNet) handle compressed images better than other  

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## Contact
For inquiries or collaborations, reach out via [todmount@gmail.com].
