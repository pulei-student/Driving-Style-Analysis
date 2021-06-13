# Driving-Style-Analysis

Hello, this is the graduation work of Herbert Gao. 

This work is mainly about recovering the article "Driving Style Analysis Using Primitive Driving Patterns With Bayesian Nonparametric Approaches" written by Wenshuo Wang, etc. Besides, this work also includes comparing analysis results with the method of directly going through "Threshold Segmention", which is done without driving pattern analysis.

To get this work done by yourself, you need to follow the steps:

## Get data.
   Get data directly from the Baidu Netdisk. Copy this website https://pan.baidu.com/s/1KlDcO_wmKEWInvukWsLyAQ, the keyword is *Hang* .
   The file *car-following* includes all drivers' car-following scenarios data extracted from SPMD dataset using Traffic-Net.
   
   **Or**
   You can try to extract data from SPMD by yourself. You will need MySQL related software to do this. 
   To do this,
   1. Download data file *DataWsu,DataFrontTarget* from SPMD website https://catalog.data.gov/dataset/safety-pilot-model-deployment-data.
   2. Use SQL software to process data with the code in file *MySQL_Code*. You can get a .csv file from SQL, which is also included in the Baidu Netdisk.
      Then process the .csv file using *Python* with the code in *Python_code/Data_processing/Car-following.py*. 
      After that, you will get the exact data file as the first method.
## Experiments.
   **You should get pyhsmm,fitter,matplolib and the related python library installed in your python interpreter.**
   1. Fitting all data to get threhold of segmention. ----code:  *Python_Code/Data_processing/Fitting_data*
   2. Get through the whole work. ----code:  *Python_Code/Processing_one_event* or *Python_Code/Processing one_person*
   3. Comparing different drivers' driving style. ----code:  *Python_code/Processing_multi_differ*
   4. Drawing Beta,Gamma distribution figure. ----code:  *draw_beta.py* and *draw_gamma.py*
