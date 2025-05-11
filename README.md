
# Identifying Frail Older Adults in Long-Term Care Facilities Using Machine Learning on Gait and Daily Physical Activity Data from a Single Accelerometer

<div align="center">
  <img src="https://github.com/xzheng93/Frailty_Identification/blob/main/fig/data_pipeline.jpg" alt="Data Pipeline" width="500"/>
</div>
This repository contains the python code for using ML to identify Frailty in long term care facilities based on gait and physical activity derived from one accelerometer as presented in[Identifying Frail Older Adults in Long-Term Care Facilities Using Machine Learning on Gait and Daily Physical Activity Data from a Single Accelerometer].



### main_loo_comparison.py: 
Demonstrates the performance comparison across different machine learning models using a leave-one-out (LOO) approach.

### main_loo.py: 
Calculates SHAP values based on the LOO approach. Although the XGBoost model is retrained in this script, we ensured the same performance as in main_loo_comparison.py by carefully setting the random seed.
