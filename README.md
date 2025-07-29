
#  Brain Tumor Classification using MRI Images

This project applies Deep Learning techniques to classify brain MRI images into one of four categories:
**Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.  
MRI-based tumor classification assists in early diagnosis and better treatment planning.

## Project Structure


```bash
PROJECT\_FINALS/
│
├── data/
│   ├── raw/                        # Original MRI images
│   │   ├── Training/
│   │   └── Testing/
│   └── processed/                  # Preprocessed and saved NumPy arrays
│       ├── X\_train.npy
│       ├── y\_train.npy
│       ├── X\_val.npy
│       ├── y\_val.npy
│       ├── X\_test.npy
│       └── y\_test.npy
│
├── notebooks/                     # Jupyter Notebooks for experiments
│   ├── 01\_data\_exploration.ipynb
│   └── 02\_prepressed\_base\_model.ipynb
│
├── outputs/                       # Trained model files
│   ├── base\_model.keras
│   └── MobileNetV2\_model.keras
│
├── src/                           # Source code and utilities
│   ├── data\_utils.py
│   ├── preprocess\_image.py
│   ├── utils\_evaluation.py
│   └── visualize\_utils.py
│
└── README.md                      # Project documentation
```

## 🚀 Project Objectives

- Load and explore the brain tumor dataset
- Preprocess MRI images (resizing, normalization)
- Visualize class distribution and sample images
- Train a baseline CNN model from scratch
- Apply **Transfer Learning** using MobileNetV2
- Evaluate models using metrics: Accuracy, Precision, Recall
- Save the best model for future predictions

---

##  Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Matplotlib, Scikit-learn
- Jupyter Notebook

Install dependencies:
```bash
pip install -r requirements.txt
```


##  Dataset Info

* Total Images: 7023
* Classes:

  * Glioma Tumor
  * Meningioma Tumor
  * Pituitary Tumor
  * No Tumor
* Source: Combined from Figshare, SARTAJ, and Br35H datasets



##  Results Summary

| Model          | Val Accuracy | Test Accuracy |
| -------------- | ------------ | ------------- |
| Baseline CNN   | 77%          | \~76%         |
| MobileNetV2 TL | **94.2%**    | **94.2%**     |




