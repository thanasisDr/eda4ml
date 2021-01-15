# eda4ml : Python library for EDA on ML projects

## Package Description

eda4ml is a Python module for Explorative Data Analysis (EDA) required in the machine learning process, built on top of Pandas, Seaborn, Scipy and Matplotlib
Inspired by the amazing kaggle notebook 'Comprehensive data exploration with Python' by Pedro Marcelino.
source: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python



## Installation

Dependencies
eda4ml requires:

Python
Pandas 
NumPy 
Seaborn 
SciPy 
Matplotlib 


### User Installation

If you already have a working installation of the dependencies packages, the easiest way to install eda4ml is using pip

`pip install eda4ml`


### Source Code

You can check the latest sources with the command:

`git clone https://github.com/thanasisDr/eda4ml.git`


## Features - methods

 - **describe**: Provides descriptive data for both numerical and categorical features

 - **numeric_col**: Provides the numerical features of the dataset

 - **categorical_cols**: Provides the categorical features of the dataset

 - **discrete_possible_cols**: Provides the numerical features of the dataset that get up to k discrete values 

 - **categorical_cols_for_dummies**: Provides the categirical features which have limited cardinality add could be converted to dummy variables

 - **cols_with_missing_data**: Provides the top_N features with the highest percentage and actual size of missing data. Columns with percentage
                           higher than 15% could be removed from the dataset

 - **cols_for_imputation**: Provides the features with missing data (less than 15%) that could be imputed. Another option is to remove the entries 
                        that contain this missing data
    
 - **correlation_plot**: Provides the correlation heatmap between the featues of the dataset when it is invoked without target column 
                     Plots the top_N most or less correlated to the target column features when the target argument is defined

 - **histograms**: Plots the histograms of a list of features

 - **boxplots**: Plots the boxplots of a list of categorical features with the target variable

 - **scatterplots**: Plots the scatterplots of a list of numerical features with the target variable

 - **normality_check**: Plots checking the normality of a feature

### Example 

    import pandas as pd
    from eda4ml.edaViz_tabular import edaViz_tab
    
    df_train = pd.read_csv('train.csv') 
    
    df_eda = edaViz_tab(df_train)
    print(df_eda.cols_with_missing_data())
    df_eda.correlation_plot()`

