# NSCLC Survival Analysis

## Overview

This project applies deep learning techniques to predict survival outcomes for non-small cell lung cancer (NSCLC) patients using 3D CT scan data and clinical annotations. The model leverages a modified 3D ResNet-18 architecture and is trained to classify patients into survival categories based on the `deadstatus.event` variable.

The main highlights of the repository include:
- A comprehensive **notebook** containing all the analysis, preprocessing, model training, evaluation, and results visualization.
- A **scientific report** summarizing the methodology, results, and future directions.
- Code for preprocessing, training, and evaluating the model, allowing users to reproduce and experiment with the findings.

## Dataset

The dataset used is the **NSCLC Radiomics Lung1 dataset**, available from The Cancer Imaging Archive (TCIA). It contains:
- **3D CT scan data**
- **Clinical annotations** with the target variable `deadstatus.event`:
  - `1`: Deceased
  - `0`: Alive

**Download Link**: [NSCLC Radiomics Dataset](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)

### Instructions for Dataset Setup

1. Download the dataset from the provided link.
2. Place the downloaded files in a folder named `data` in the root directory of the project.

# Directory Structure

NSCLC-Survival-Analysis/
  - data/
    - manifest-1603198545583/
      - NSCLC-Radiomics-Lung1/
        - LUNG1-001/
        - LUNG1-002/
        - …
    - clinical_data.csv
  - src/
    - dataset.py          # Data management and preprocessing
    - model.py            # Definition of the ResNet-18 3D model
    - evaluate.py         # Script to evaluate the model
    - main.py             # Main file to train or evaluate the model
  - notebook/
    - analysis_notebook.ipynb  # Main notebook containing the full pipeline
  - report/
    - final_report.pdf    # Final report
    - figures/            # Folder containing the report’s figures
  - requirements.txt      # List of required Python packages
  - README.md

## Key Files in the Repository

### 1. **Notebook** (`notebook/NSCLC_Survival_Analysis.ipynb`)
- **Purpose**: The notebook is the most important file in the repository. It contains the entire pipeline:
  - Data preprocessing and augmentation
  - Model training and evaluation
  - Results visualization (ROC curve, confusion matrix, probability distributions, etc.)
  - Threshold optimization
- **Usage**: Open the notebook in Jupyter or any compatible environment and follow the cells step by step. It is designed to provide a comprehensive overview of the methodology and results.

### 2. **Report** (`report/NSCLC_Survival_Analysis_Report.pdf`)
- **Purpose**: This is the final scientific report summarizing the entire project. It includes:
  - Problem description
  - Dataset details
  - Methodology
  - Results and analysis
  - Discussion and conclusions

### 3. **Source Code** (`src/`)
- This folder contains supporting scripts for data preprocessing, model definition, and evaluation.
  - **`datasets.py`**: Contains code for loading and preprocessing the NSCLC dataset.
  - **`model.py`**: Defines the 3D ResNet-18 architecture used in the project.
  - **`main.py`**: The primary script to evaluate the trained model and generate a confusion matrix.
    - **Usage**:
      - To evaluate the model: 
        ```
        python src/main.py --mode evaluate
        ```
      - Ensure that the dataset is correctly placed and that the required dependencies are installed.

### 4. **Results and Outputs**
- The notebook also generates key plots and outputs:
  - **ROC curve**
  - **Confusion matrix**
  - **Distribution of predicted probabilities**
  - **Threshold-optimized confusion matrix**

## How to Run the Code

1. **Setup Environment**:
   - Ensure you have Python 3.8 or later installed.
   - Install the required packages by running:
     ```
     pip install -r requirements.txt
     ```

2. **Prepare the Dataset**:
   - Download the dataset and place it in the `data` folder as described above.

3. **Run the Notebook**:
   - Open `notebook/NSCLC_Survival_Analysis.ipynb` in Jupyter Notebook or JupyterLab and execute the cells step-by-step.

4. **Evaluate the Model**:
   - Use the `main.py` script to evaluate the model and generate a confusion matrix. Run:
     ```
     python src/main.py --mode evaluate
     ```

## Notes
- The current pipeline has been optimized for the NSCLC dataset but may require further adaptation to generalize to other datasets.
- Ensure you have sufficient computational resources, as training and evaluation involve 3D convolutional operations that are memory-intensive.
- The results in the notebook highlight the importance of threshold optimization and probability calibration in improving classification performance.

## Future Directions
For extended analysis or to contribute:
- Experiment with additional datasets for validation.
- Explore advanced architectures like 3D DenseNets or attention-based models.
- Implement hyperparameter tuning and further optimize probability calibration techniques.

