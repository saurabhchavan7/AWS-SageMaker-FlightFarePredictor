# Flight Fare Predictor: End-to-End Machine Learning Project using AWS SageMaker

## Introduction

This project demonstrates the process of building, training, and deploying a machine learning model for predicting flight fares using AWS SageMaker. The project encompasses the entire workflow, from data preprocessing to deploying a web application. The goal is to create an effective and efficient machine learning pipeline that can handle large datasets, optimize hyperparameters, and provide accurate predictions.

### Web Application

Check out the deployed web application here: [Flight Fare Predictor](https://aws-sagemaker-flightfarepredictor-mridrukbwuejehfenkku72.streamlit.app/)


## Project Workflow

### 1. AWS SageMaker Overview

- **AWS SageMaker**: A fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly.
- **S3**: Amazon Simple Storage Service (S3) is used for storing data.
- **EC2**: Amazon Elastic Compute Cloud (EC2) provides the compute resources for training models.
- **IAM**: AWS Identity and Access Management (IAM) is used for managing access to AWS services and resources securely.

### 2. Setting up the Environment

- **Setting up AWS environment and SageMaker instance**: Configured the AWS environment, set up SageMaker instances, and established the necessary permissions using IAM roles.
- **GitHub Setup**: Initialized a local and remote GitHub repository for version control and collaboration.

### 3. Data Cleaning

- **Data Cleaning using Numpy and Pandas**: Implemented best practices for data cleaning to ensure the dataset is free of inconsistencies and ready for analysis.

### 4. Exploratory Data Analysis (EDA)

- **Understanding datasets**: Conducted a systematic analysis of the dataset to understand its structure and contents.
- **Plots and Statistical Measures**: Created various plots and calculated statistical measures to gain insights into the data.
- **Hypothesis Testing**: Performed hypothesis tests to validate assumptions and draw meaningful conclusions from the data.

### 5. Feature Engineering

- **Feature Engineering Techniques**: Applied various techniques to create new features from existing data.
- **Custom Classes and Functions**: Developed scikit-learn compatible custom classes and functions for feature engineering.
- **Advanced scikit-learn Features**:
  - **Pipeline**: Streamlined the process of transforming data and applying models.
  - **Feature Union**: Combined multiple feature extraction methods.
  - **Function Transformer**: Applied custom transformations.
  - **Column Transformer**: Applied different preprocessing steps to different subsets of features.

### 6. Model Training and Hyperparameter Tuning

- **Preprocessing Data**: Preprocessed training and validation datasets and uploaded them to S3 buckets.
- **Setting up ML Model**: Configured an XGBoost model in SageMaker.
- **Hyperparameter Tuning**: Used SageMaker's hyperparameter tuning capabilities to find the best model configuration.
- **Training and Tuning**: Trained the model using EC2 instances and tuned it to improve performance.
- **Saving the Model**: Saved the best model to an S3 bucket for later use.

### 7. Model Evaluation

To evaluate the performance of the trained XGBoost model, we used the R² score metric. The R² score measures the proportion of the variance in the dependent variable that is predictable from the independent variables. 

The evaluation results were as follows:
- **Training Set R² Score**: 0.6586
- **Validation Set R² Score**: 0.6156
- **Test Set R² Score**: 0.5925

These scores indicate how well the model generalizes to unseen data, with a higher R² score representing a better fit.

### 8. Web Application Development

- **Creating a Web Application**: Developed a web application using Streamlit to interact with the model and make predictions.
- **Deployment**: Deployed the web application using Streamlit Cloud to make it accessible to users.

## Conclusion

The project showcases the full lifecycle of a machine learning project, from data preprocessing to model deployment. By leveraging AWS SageMaker and other AWS services, the project demonstrates how to handle large datasets, perform hyperparameter tuning, and deploy a machine learning model in a scalable and efficient manner.


## References

- **AWS SageMaker Documentation**: [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- **Streamlit Documentation**: [Streamlit](https://docs.streamlit.io/)
