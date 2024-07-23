## Overview

The objective of this project is to identify gene expression patterns associated with different conditions or diseases, leveraging advanced data processing and model training techniques. RNA sequencing (RNA-Seq) is a teachnique used to quantify and analyze gene expression levels across different conditions or samples. The classification of RNA-Seq data can be used to identify which genes are differently expressed between healthy and diseased samples, or between different diseased states, thus aiding in the diagnosis, treatment, and understanding of various diseases and conditions.

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [**References**](#references)

## Dataset

The dataset used in this project:

- **Source**: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

- **Description**: This dataset is part of the RNA-Seq (HiSeq) PANCAN data set. It is a random extraction of gene expressions of patients having different types of tumor: BRCA (Breast invasive carcinoma), KIRC (Kidney renal clear cell carcinoma), COAD (Colon adenocarcinoma), LUAD (Lung adenocarcinoma) and PRAD (Prostate adenocarcinoma). 

  The downloaded dataset file TCGA-PANCAN-HiSeq-801x20531.tar.gz contains two CSV files. 

  - **data.csv**: Contains the feature set (X) representing gene expression data. 
  - **labels.csv**: Contains the target labels (y) corresponding to the samples in the feature set.

## Project Structure

The outline of the project repository:

```
RNASeq-GeneExpression-ML/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── split_data
│   │   ├── transformed
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
├── src/
│   ├── load_data.py
│   ├── split.py
│   ├── smote.py
│   ├── pca.py
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── naive_bayes.py
│   │   ├── knn.py
│   │   ├── mlp.py
│   │   ├── gradient_boosting.py
│   │   ├── ada_boost.py
│   │   ├── xgboost.py
│   │   ├── lightgbm.py
│   │   ├── catboost.py
│   ├── evaluate_models.py
├── results/
│   ├── logistic_regression.txt
│   ├── decision_tree.txt
│   ├── random_forest.txt/
│   ├── svm.txt/
│   ├── naive_bayes.txt/
│   ├── knn.txt/
│   ├── mlp.txt/
│   ├── gradient_boosting.txt/
│   ├── ada_boost.txt/
│   ├── xgboost.txt/
│   ├── lightgbm.txt/
│   ├── catboost.txt/
├── README.md
├── requirements.txt
```

## Running the Project

1. **Clone the repository:**

```
git clone git@github.com:shalinis602/RNASeq-GeneExpression-ML.git
cd RNASeq-GeneExpression-ML
```

2. **Create and activate a virtual environment:**
```
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install the dependencies:** 
```
pip install -r requirements.txt
```

4. **Fast load the dataset and verify its contents:**

```
python src/load_data.py
```

5. **Split the data into training, testing and validation sets:**

```
python src/split.py
```

6. **Preprocess the dataset:**

```
python src/smote.py
python src/pca.py
```

7. **Train the models:**

```
python src/models/random_forest.py
```

8. **Evaluate the models:**

```
python src/model_evaluation.py
```

Alternatively, you can run the Jupyter notebooks located in the `notebooks/` directory to interactively execute the code.

## Models Used

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **k-Nearest Neighbors (KNN)**
- **Multilayer Perceptron (MLP)**
- **Gradient Boosting**
- **AdaBoost**
- **XGBoost**
- **LightGBM**
- **CatBoost**

## Results

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## Contributing

If you find any issues or have suggestions for improvements or expanding the project, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## **References**
1. [RNA-Seq Gene Expression Classification Using Machine Learning Algorithms](https://ernest-bonat.medium.com/rna-seq-gene-expression-classification-using-machine-learning-algorithms-de862e60bfd0) 
