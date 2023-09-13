# My Solution for the ICR - Identifying Age-Related Conditions Kaggle Competition

This GitHub repository contains my solution for the Kaggle competition titled "ICR - Identifying Age-Related Conditions."

## The Competition
The goal of this competition is to predict whether a person has one or more of three medical conditions (Class 1) or none of these conditions (Class 0) based on health characteristic measurements. This competition aims to enhance existing methods for predicting medical conditions and advance the field of bioinformatics.

## Goals
 - Participate in and gain experience from a Kaggle competition for the first time.
 - Formulate a model for a problem outside my domain of expertise.
 - Utilize Optuna for hyperparameter optimization of models.
 - Consider stacking ensemble models as a final approach if appropriate.

## Solution Overview
Starting with the bottom line: The performance of the final stacked models is subpar. The primary reason for the suboptimal performance is overfitting, primarily caused by the small database, which differs significantly from the final test set. In hindsight, the most critical improvement that could have been made was conducting a more thorough data exploration and analysis.

To create the model, I employed XGBoost and random forest models with hyperparameters optimized using Optuna. The models were trained on several stratified folds of the data and subsequently merged to produce a single prediction.

## Code Overview

1. **main.ipynb**
   - **Data Preparation & Analysis:** This section covers data loading and preprocessing to ensure that the data is in a suitable format for model training.
   - **Model Training Using Optimized Parameters:** The code includes the training of machine learning models using the prepared data.
   - **Evaluation:** The evaluation section demonstrates how the model's performance is assessed, calculating a balanced logarithmic loss to evaluate the predictions.
   - **Submission:** This part of the code generates the submission file required by the Kaggle competition, formatting the predictions as specified.

2. **opt.py** and **opt-xgb.py**
   - Similar to the **main.ipynb** notebook but focused on hyperparameter optimization using Optuna.

3. **opt.sh** and **opt-xgb.sh**
   - Bash scripts that run the optimization files in the background.

4. **parse_log.ipynb**
   - Post-processing files related to hyperparameter optimization.

### Used Packages:
 - Scikit-Learn
 - XGBoost
 - Optuna
 - Pandas
 - NumPy
 - Seaborn
 - Matplotlib
