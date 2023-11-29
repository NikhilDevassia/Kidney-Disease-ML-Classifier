# Kidney-Disease-ML-Classifier
This machine learning project aims to detect kidney cancer from X-ray images using a VGG16-based classifier. The project is designed with a modular structure and implemented in Python, covering an end-to-end pipeline that includes data preprocessing, model creation, training, deployment using Streamlit, model monitoring with MLflow, and pipeline tracking with DVC.

### Working Demo 
##### Train
https://github.com/NikhilDevassia/Kidney-Disease-ML-Classifier/assets/102344001/31d87a2a-2f57-4e82-869d-18626942093f
#### Predict
[kidney disease predict.webm](https://github.com/NikhilDevassia/Kidney-Disease-ML-Classifier/assets/102344001/a5528aaa-bb4a-4644-99c6-88dce29125e1)
# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/NikhilDevassia/Kidney-Disease-ML-Classifier.git
```
### STEP 01- Create a conda environment or python env after opening the repository
##### Conda env
```bash
conda create -n env python=3.9 -y
```

```bash
conda activate env
```
##### Python env
```bash
python3 -m venv env 
```

```bash
source/evn/bin/activate
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
streamlit run app.py
```
