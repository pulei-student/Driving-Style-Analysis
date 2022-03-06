# Driving-Style-Analysis

**Please notice that the version 2.0 of this work has been online, please check https://github.com/Herbert-Gao/Revision-2.0-Driving-Style-Analysis-Using-Drving-Patterns-with-Bayesian-Approaches to see more info. After 1 month, this repo will be used for gathering all my recent work in driving analysis.**

Hello, this is the graduation work of Herbert Gao. 

This work is mainly about recovering the article "Driving Style Analysis Using Primitive Driving Patterns With Bayesian Nonparametric Approaches" written by Wenshuo Wang, etc. Besides, this work also includes comparing analysis results with the method of directly going through "Threshold Segmention", which is done without driving pattern analysis.

### To get this work done by yourself, you need to follow the steps:

## Get data.
   Get data directly from the Baidu Netdisk. Copy this website 链接：https://pan.baidu.com/s/1azR5h-iFFpfMmE8tEvu_ww?pwd=275g 
, the keyword is [**275g**] .
   The file *car-following* includes all drivers' car-following scenarios data extracted from SPMD dataset using Traffic-Net.
   
*Or*
  
   You can try to extract data from SPMD by yourself. You will need MySQL related software to do this.The recommended software is MySQL workbench. 
   To do this,
   1. Download data files *DataWsu,DataFrontTarget* from SPMD website https://catalog.data.gov/dataset/safety-pilot-model-deployment-data.
   2. Use your MySQL software to process data with the code in file *MySQL_Code*. Please follow the memo in the file. Differet softwares matter! Finally, you can get a .csv file after processing the whole data. Check the fromat of your data profile with mine downloaded from BaiduNetdisk to see weather you have done rightly.
      Then process the .csv file using *Python* with the code in *Python_code/Data_processing/Car-following.py*. 
      After that, you will get the exact data file as the first method.

## Experiments.
   **You should get numpy,pybayesics,future,pyhsmm,fitter,matplolib and the related python library installed in your python interpreter. Mind that I have done some changes in pyhsmm to have a better vision of the results.**
   
   **Download my pyhsmm library. Get into it. Input [pip install -e.] in your comand window.**
   1. Fitting all data to get threhold of segmention. ----code:  *Python_Code/Data_processing/Fitting_data*
   2. Get through the whole work. ----code:  *Python_Code/Processing_one_event* or *Python_Code/Processing one_person*
   3. Comparing different drivers' driving style. ----code:  *Python_code/Processing_multi_differ*
   4. Drawing Beta,Gamma distribution figure. ----code:  *draw_beta.py* and *draw_gamma.py*

## Results.
You can comparing your results with mine in the file *all_driver_results* and *all_driver_noseg*.

## Plus
SimHei.ttf is kind of font that you may need in drawing figures.
