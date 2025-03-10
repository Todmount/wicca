# WICCA : Wavelet-based Image Compression & Classification Analysis.
*A framework for exploring wavelet compression effects on image classification.*

## 📌 Overview
WICCA is a research-driven framework that investigates how wavelet-based image compression affects classification performance. The primary focus is on reducing high-resolution images (>2000px) into compact representations while maintaining classification accuracy. Currently, the framework is centered on Haar wavelet compression.

## 🎯 Project Goals
### 🌍 Global Goal
- Evaluate the impact of various wavelet compression techniques on different classifiers.

### 🔍 Current Goal
- Analyze how Haar wavelet compression influences classification performance.
- Compare classification results between original high-resolution images and their compressed counterparts.

## 🛠 How It Works
1. **Dataset Preparation**: Large images (>2K resolution) are preprocessed.
2. **Wavelet Compression**: Images are compressed using Haar wavelet transform.
3. **Classification**: Pre-trained classifiers are applied to both the original and compressed datasets.
4. **Performance Analysis**: Results are compared to assess accuracy trade-offs.

## 🚀 Features
✅ Supports large image datasets  
✅ Implements wavelet-based image compression  
✅ Works with multiple classifiers  
✅ Provides performance comparison & analysis tools  

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/WCLC.git
cd WCLC

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
- [ ] Add a side-by-side visualization of the original image and its icon
- [ ] Implement minimal Flask MVP
- [ ] Dockerize the project
- [ ] Extend to other wavelets (Daubechies, Coiflet, etc.)
- [ ] Integrate deep learning-based classifiers
- [ ] Optimize for large-scale datasets

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 Lic
This project is licensed under the MIT License.

## Contact
For inquiries or collaborations, reach out via [todmount@gmail.com].
