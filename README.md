# Modeling kp for oxidation of superalloy
By [Fan Yang](https://github.com/fanYang-X), [Wenyue Zhao](https://shi.buaa.edu.cn/09652/zh_CN/index.htm).
The work is in the review process of the SCI journal.

## Background  
In superalloys, oxidation has an erosive effect on the internal microstructure, and the degradation of γ' strengthened phase caused by oxidation has a significant effect on the mechanical properties of superalloys. Based on machine learning, a prediction model of the relationship between compositions, oxidation and microstructure is constructed to evaluate the oxidation properties of new components and predict the degradation of near-surface microstructure, thus providing guidance for alloy design.

## Updata

***15/2/2023***
Initial commits:

Kp prediction model (data and code)   

   data: consist of init composition data and activity data (Thermo-Calc); the data with label (continuous) 
   
   model: Model.ipynb, the base-model based the activity of Ni, Al, Cr and the cbfv-model. 
   

## Usage 

The versions of the pyhton library used are as follows:  
pandas -- 1.3.1  
numpy -- 1.20.3  
scikit-learn -- 1.1.2  
lightgbm -- 3.2.1  
optuna -- 2.9.1
