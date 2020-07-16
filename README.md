# GIMP

# Generative Incomplete Multi-View Prognosis Predictor for Breast Cancer: GIMP

# References

Our manuscipt titled with "Generative Incomplete Multi-View Prognosis Predictor for Breast Cancer: GIMP" has been submitted to IEEE JBHI.

# Requirements
[python 3.6](https://www.python.org/downloads/)


[TensorFilow 1.12](https://www.tensorflow.org/install/)

[keras 2.2.4](https://pypi.org/project/Keras/)


[scikit-learn 0.20.0](http://scikit-learn.org/stable/)


[matplotlib 3.0.1](https://matplotlib.org/users/installing.html)



# Usage
GIMP_step1_2.py
GIMP_step3.py
cnn_clinical.py
cnn_cnv.py
cnn_exp.py
AdaboostRF.py
ttest.py

# Process to execute the GIMP architecture.

(1)  Firstly execute GIMP_step1_2.py using paired samples(METABRIC_clinical_1600_common.csv,METABRIC_cnv_1600_common.csv,METABRIC_gene_exp_1600_common.csv) and unpaired samples (METABRIC_clinical_1600.csv,METABRIC_cnv_1600.csv,METABRIC_gene_exp_1600.csv) of all the views.

(2)  After successfull run of GIMP_step1_2.py will get the generated missng samples of each view in three different csv files : cln_missing_data.csv, exp_missing_data.csv and cnv_missing_data.csv and common latent representation in common_rep.csv file.

(3)  Randomly select any samples from the generated samples and fill the missing sample values with the respective view's generated samples to create complete multi-view data. (metabric_cln_generated.csv, metabric_cnv_generated.csv, metabric_exp_generated.csv)

(4)  Now, execute GIMP_step3.py using the generated complete mult-view data (metabric_cln_generated.csv, metabric_cnv_generated.csv, metabric_exp_generated.csv) and get again three different csv files cln_missing_data.csv, exp_missing_data.csv and cnv_missing_data.csv and common latent representation in common_rep.csv file.

(5) Repeat the step (3) to get the final complete multi-view data and integrate class labels (metabric_cln_final.csv, metabric_cnv_final.csv and metabric_exp_final.csv)

(6) Run cln_clinical.py, cnn_cnv.py and cnn_exp.py for metabric_cln_final.csv,metabric_cnv_final.csv and metabric_exp_final.csv, respectively to get the hidden features in three different csv files : gatedAtnClnOutput.csv, gatedAtnCnvOutput.csv and gatedAtnExpOutput.csv

(7) Combine all the hidden features of different modalities to form stacked features : METABRIC_COMPLETE_GIMP.csv

(8) Run AdaboostRF.py and pass the stacked feature(METABRIC_COMPLETE_GIMP.csv) as input to get the final prediction output.

(9) Once final prediction has been made use ttest.py to perform statistical significance test.


# Note : Similar steps has to be followed to execute the Baseline 1, Baseline 2 and Baseline 3 architectures.






