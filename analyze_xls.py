#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


xls = pd.ExcelFile("data/big_analysis.xlsx")
section_0 = xls.parse(0)
section_1 = xls.parse(1)
print(section_0.columns)

xls = pd.ExcelFile("data/study_summer_data.xlsx")
general_data = xls.parse(0)
print(general_data.columns)

all_data_df = pd.concat([general_data, section_1], axis=1)

relevant_df = all_data_df[['PSY', 'subject_number_of_poses', 'min_matrix_error', 'mean_matrix_error']].dropna()
relevant_df = all_data_df[['GPA', 'subject_number_of_poses']].dropna()
print(relevant_df.head)

# result = sm.ols(formula="PSY ~ subject_number_of_poses + min_matrix_error + mean_matrix_error", data=relevant_df).fit()
result = sm.ols(formula="GPA ~ subject_number_of_poses", data=relevant_df).fit()
print result.summary()
sns.regplot(x='subject_number_of_poses', y='GPA', data=relevant_df)
plt.show()

