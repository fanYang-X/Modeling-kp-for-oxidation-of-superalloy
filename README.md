# Modeling kp for oxidation of superalloy
By [Fan Yang](https://github.com/fanYang-X).

## Background  
In superalloys, oxidation has an erosive effect on the internal microstructure, and the degradation of γ' strengthened phase caused by oxidation has a significant effect on the mechanical properties of superalloys. Based on machine learning, a prediction model of the relationship between compositions, oxidation and microstructure is constructed to evaluate the oxidation properties of new components and predict the degradation of near-surface microstructure, thus providing guidance for alloy design.

## Updata

***15/2/2023***
Initial commits:

1. Kp prediction model (data and code)   

   data: consist of init composition data and activity data (Thermo-Calc); the data with label (continuous) 
   model: Model.ipynb, the base-model based the activity of Ni, Al, Cr and the cbfv-model. 
   
2. Near surface microstructure prediction model (data and code)

## Usage 

The versions of the pyhton library used are as follows:  
pandas -- 1.3.1  
numpy -- 1.20.3  
scikit-learn -- 1.1.2  
lightgbm -- 3.2.1  
