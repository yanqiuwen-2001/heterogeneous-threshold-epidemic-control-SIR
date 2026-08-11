# heterogeneous-threshold-epidemic-control-SIR

This repository contains the code used for the article **"Heterogeneous Threshold Control in Multi-Patch Epidemic Models: Final-Size Optimization and Theory-Guided Target-Patch Identification"**.

## File Description

1. `Data_generation/Data_generation.py`  
   Generates the synthetic data required by the model.

2. `Fig1`--`Fig7`  
   Programs for producing Fig. 1--Fig. 7 in the article.

3. `Fig5/Multi_models.py`  
   Performs cross-validation for hyperparameter selection and compares different classification models.

4. `Fig5/BiLSTM_optimal.py`  
   Trains the BiLSTM-2d model using the optimal hyperparameters and reports the model evaluation metrics.

5. `Real_Delta`  
   Contains the case study in Section 4.4, where the trained model is applied to real epidemic data.

6. `Table1`  
   Contains the data required for Table 1 in the article.

