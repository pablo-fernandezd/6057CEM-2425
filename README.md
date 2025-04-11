# Iris Classification with Residual & Attention Networks

## Project Description

This project implements and evaluates two neural network architectures – a Residual Network (ResNet-inspired) and an Attention-based network – for classifying the Iris flower dataset. The goal is to predict the species of an Iris flower (Setosa, Versicolor, or Virginica) based on four input features: sepal length, sepal width, petal length, and petal width.

The project follows a standard machine learning workflow:
1.  **Data Loading & Preprocessing:** Loading the dataset, encoding labels, splitting into train/validation/test sets, and feature scaling.
2.  **Exploratory Data Analysis (EDA):** Visualizing data distributions, correlations, and dimensionality reduction to understand the dataset characteristics. Plots are saved.
3.  **Model Definition:** Implementing the Residual and Attention network architectures using TensorFlow/Keras. Model diagrams are potentially saved.
4.  **Hyperparameter Tuning:** Using KerasTuner (RandomSearch) to find optimal hyperparameters for both models.
5.  **Model Training:** Training the best models found during tuning with appropriate callbacks (EarlyStopping, ReduceLROnPlateau).
6.  **Model Evaluation:** Evaluating the trained models on the test set using various metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC AUC). Plots are saved.

This project was developed as part of the [Mention Course, e.g., 6057CEM module].

## Features

* Modular code structure (`utils`, `core`, `pipeline`).
* Comprehensive EDA with various plots saved to disk.
* Implementation of a custom Residual Network architecture.
* Implementation of a custom Attention mechanism within a feed-forward network.
* Generation of model architecture diagrams (HTML).
* Hyperparameter tuning using KerasTuner.
* Detailed model evaluation with results and visualizations saved to disk.
* Jupyter Notebook (`main.ipynb`) demonstrating the end-to-end workflow.

## Technologies Used

* Python 3.11
* TensorFlow / Keras
* KerasTuner
* Scikit-learn
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Conda (for environment management)

## Project Structure
scientificProject/
├── project/
│   ├── core/                 # Core model architecture definitions
│   │   ├── attention_model.py
│   │   └── residual_model.py
│   ├── data/                 # (Likely contains saved datasets, e.g., train/test splits)
│   ├── graphs/               # Contains all generated visualizations
│   │   ├── eda/              # Plots generated during Exploratory Data Analysis
│   │   ├── model_analysis/   # Plots generated during model evaluation (e.g., confusion matrices, ROC)
│   │   └── model_diagrams/   # HTML files visualizing the Keras model architectures
│   ├── pipeline/             # Training and tuning pipeline scripts
│   │   ├── training_framework.py
│   │   ├── training_framework_attention.py # (Likely helper for tuner)
│   │   └── training_framework_residual.py  # (Likely helper for tuner)
│   ├── utils/                # Utility scripts for data processing, EDA, evaluation
│   │   ├── data_processing.py
│   │   ├── eda_module.py
│   │   └── evaluation_metrics.py
│   ├── attention_tuning/     # KerasTuner results for Attention model
│   ├── residual_tuning/      # KerasTuner results for Residual model
│   ├── main.ipynb            # Main Jupyter Notebook for workflow execution
│   └── pycache/          # (Python cache - can be ignored/deleted)
├── environment.yml           # Conda environment definition
└── README.md                 # This file

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd scientificProject
    ```
2.  **Create and activate the Conda environment:**
    Ensure you have Anaconda or Miniconda installed. The `project/graphs` and `project/data` directories might need to be created manually if they don't exist after cloning and before running the notebook, although the scripts might create `graphs` subdirectories automatically when saving plots.
    ```bash

    conda env create -f environment.yml
    conda activate ann
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open and run the cells in `project/main.ipynb`.

## Usage

Navigate through the `project/main.ipynb` notebook. Run the cells sequentially to perform data loading, EDA, model definition, hyperparameter tuning (this might take some time), training, and evaluation. Generated plots and model diagrams will be saved into the `project/graphs/` subdirectories. Processed data might be saved in `project/data/`. Results and visualizations will also be displayed within the notebook.

## Results

The final performance metrics (accuracy, precision, recall, F1-score, confusion matrices) for both the Residual and Attention models on the test set are presented in the `main.ipynb` notebook. Visualizations supporting the results (EDA plots, evaluation metrics plots, model diagrams) are saved in the `project/graphs/` directory.

