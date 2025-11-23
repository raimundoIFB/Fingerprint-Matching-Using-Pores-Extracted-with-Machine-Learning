# Fingerprint-Matching-Using-Pores-Extracted-with-Machine-Learning
The main purpose of this project is to train and compare the performance of two deep learning models (U-Net and ResNet) on the L3-SF dataset for specific pore prediction metrics. This repository contains the base implementation for this research. 

# Usage

1 - Clone this repository into your own machine <br>```git clone https://github.com/ciromoraesr/Fingerprint-Matching-Using-Pores-Extracted-with-Machine-Learning.git```<br><br>
2 - Download the L3-SF dataset in https://andrewyzy.github.io/L3-SF/ <br><br>
3 - Create your python enviroment, then install the requirements with ```pip install -r requirements.txt ``` <br><br>
4 - Run the main function with ``` python main.py ``` <br><br> 
5 - After generating every model and every desired plot. Run the fingerprint matching algorithm using ``` python new_matcher.py ```<br><br># Fingerprint Matching Using Pores Extracted with Machine Learning

## 📄 Official Publication

**THEODORO, Ciro Luís; VASCONCELOS, Raimundo C. S.**  
*Fingerprint Matching Using Pores Extracted with Machine Learning.*  
XXXVIII Conference on Graphics, Patterns and Images — WTG 2025, Salvador.  
Anais Estendidos da SIBGRAPI 2025. SBC, 2025. pp. 291–294.
<https://sol.sbc.org.br/index.php/sibgrapi_estendido/article/view/38317>

Please cite this publication if you use this repository in academic work.

---

## 🧠 Method Overview

The system implements a **five‑stage fingerprint recognition pipeline**:

```
[1] Image Preprocessing  
        ↓
[2] Pore Extraction (Deep Learning)  
        ↓
[3] Post-Processing & Coordinate Refinement  
        ↓
[4] Pore-Based Matching  
        ↓
[5] Evaluation & Validation
```

### Highlights
- Extracts pores using a trained neural network architecture.
- Performs upsampling and filtering to enhance pore visibility.
- Matches fingerprints using geometric pore configurations rather than minutiae.
- Robust under noise, partial prints, and low-quality data.

---

## 📁 Repository Structure

```
├── architecture2.py       # Neural network definition for pore detection
├── coord_extractor.py     # Converts model output into pore coordinates
├── data.py                # Dataset loading and handling
├── main.py                # Main pipeline: preprocessing → extraction → matching
├── new_matcher.py         # Novel pore-based matching algorithm
├── process.py             # Image enhancement and normalization
├── tests.py               # Validation routines for the full pipeline
├── upsample.py            # Upsampling methods for high-res pore detection
└── README.md              # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```
git clone <repository-url>
cd Fingerprint-Matching-Using-Pores-Extracted-with-Machine-Learning
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

If you prefer, I can generate this file automatically.

---

## ▶️ How to Use

### Run the full pipeline
```
python main.py
```

### Run tests
```
python tests.py
```

---

## 📊 Applications

- Forensics & criminal identification  
- High‑security access systems  
- Fingerprint analysis under low‑quality conditions  
- Academic research on pore‑level biometrics  

---

## 📬 Contact & Contributions

Contributions, suggestions, and pull requests are welcome.  
This project can be extended with:
- Model retraining scripts  
- Dataset integration tools  
- Evaluation metrics dashboard  
- Visualization utilities for pore maps  


# Fingerprint-Matching-Using-Pores-Extracted-with-Machine-Learning
# Fingerprint-Matching-Using-Pores-Extracted-with-Machine-Learning
