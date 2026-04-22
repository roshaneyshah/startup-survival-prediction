# Startup Survival Prediction Using Logistic Regression

## Student Names and Registration Numbers
- Roshane Shahbaz

## Problem Statement
This project focuses on developing a **startup survival prediction system** using classical machine learning techniques. The primary objective is to design, optimize, and evaluate a **Logistic Regression** model for accurate classification of startups into survival or closure based on historical company and funding data, while treating it as a strong baseline model.

## Tools and Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Feature Extraction Techniques
- Aggregation of funding data (total funding, average round size, maximum round size)
- Time-based features (days to first funding, funding duration)
- Encoding categorical variables (country and industry)
- Grouping rare categories into OTHER
- Median-based handling of missing values

## NLP Models
### Primary Model
- **Logistic Regression**

### Comparative Models
- Not included (single model baseline approach)

## How to Run the Code
- Install the required Python libraries listed in the `requirements.txt` file.
- Place the dataset files in the specified `data` directory following the folder structure used in the project.
- Run preprocessing and feature engineering scripts to generate structured features.
- Execute the training and evaluation script. The **Logistic Regression** model is trained using a pipeline with imputation and scaling.
- Review evaluation metrics and results saved in the `results` directory.

## Dataset Source
The dataset consists of labeled startup ecosystem data including companies, funding rounds, acquisitions, and IPO records. The dataset used in this project is available at: https://www.kaggle.com/datasets/justinas/startup-investments/data

## Project Structure
- `data` - contains dataset references and folder structure
- `features` - contains extracted feature files
- `models` - contains training and evaluation scripts
- `results` - contains evaluation metrics and visualizations
- `presentation` - contains project slides
- `report` - contains the final project report

## Notes
- All code files are properly commented to ensure clarity and reproducibility.
- The **Logistic Regression** model is treated as the main model, with class imbalance handled using balanced class weights.
- Feature preprocessing is implemented using a pipeline including imputation and scaling.
- The model outputs probability scores for evaluation using **ROC-AUC**.
