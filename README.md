# PD-CR
Here we provide a python code for the "Primal-Dual for Classification with Rejection (PD-CR)" method.
This method works on Spyder 3.3.6.
Users will need to have some experience in coding to use the scripts.
The code is currently usable with ".csv" files following a specific organization (see Test datasets for an example). Users may need to modify the scripts to use PD-CR on new files.

"Functions.py" contains the functions used in the main scripts.

Scipt "PD-CR_vs_PLSDA_and_RF.py" will compare the PD-CR classification method to PLSDA and Random Forests in terms of accuracy (using cross validation) and feature selection. 
- The accuracies for each method are available in the output "accTestCompare". 
- The details concerning the accuracy of PD-CR in each class are available in the output "acctest"
- The selected features and their weights for the classification for each method are available in the output "df_featureList".

"Script_rhoComputing.py" will give the result of the prediction performed with PD-CR for each sample using cross validation. 
- The output "df_confidence" is a table comparing the predicted label for each sample "Ypred" to the original label "Yoriginal" and the confidence score for the prediction (CSP) "rho".
- The histogram of the CSP for every sample, the curve representing the false discovery rate depending on the CSP threshold for classification with rejection and the curve representing the rejected samples rate depending on the CSP threshold for classification with rejection are also provided.
