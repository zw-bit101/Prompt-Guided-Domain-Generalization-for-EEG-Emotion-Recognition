# Prompt-Guided-Domain-Generalization-for-EEG-Emotion-Recognition
Codes for the paper Prompt-Guided Domain Generalization for EEG  Emotion Recognition.
## Installation
* Python 3.7.12
* Pytorch 1.13.1
* NVIDIA CUDA 11.2
* Numpy 1.21.6
* Scikit-learn 1.0.2
* scipy 1.7.3
  
## Preliminaries
* Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)
  
# Training 
* PGDG model definition file: networks.py
* Pipeline of the PGDG: main_1_2_3.py

# Usage
* After modify setting (path, etc), just "python main_1_2_3.py"
