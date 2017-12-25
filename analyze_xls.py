#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

def analysis_1():
    xls = pd.ExcelFile("data/big_analysis.xlsx")
    section_0 = xls.parse(0)
    section_1 = xls.parse(1)
    print(section_0.columns)

    xls = pd.ExcelFile("data/study_summer_data.xlsx")
    general_data = xls.parse(0)
    print(general_data.columns)

    all_data_df = pd.concat([general_data, section_0], axis=1)

    relevant_df = all_data_df[['GPA', 'subject_number_of_poses', 'min_matrix_error', 'mean_matrix_error']].dropna()
    # relevant_df = all_data_df[['GPA', 'subject_number_of_poses']].dropna()
    print(relevant_df.head)

    result = sm.ols(formula="GPA ~ subject_number_of_poses + min_matrix_error + mean_matrix_error", data=relevant_df).fit()
    # result = sm.ols(formula="GPA ~ subject_number_of_poses", data=relevant_df).fit()
    print result.summary()
    # sns.regplot(x='subject_number_of_poses', y='GPA', data=relevant_df)
    # plt.show()

def analysis_2():
    xls = pd.ExcelFile("data/all_data.xlsx")
    data_df = xls.parse(0)

    data_clean = data_df.drop(data_df[data_df.step_id == 8].index)
    data_clean = data_clean.drop(data_clean[data_clean.step_id == 0].index)

    result = sm.ols(formula="number_of_poses ~ step_id + C(matrix) -1", data=data_clean).fit()
    print(result.summary())

def analysis_3():
    xls = pd.ExcelFile("data/big_analysis.xlsx")
    section_0_df = xls.parse(0)
    section_1_df = xls.parse(1)
    section_2_df = xls.parse(2)

    section_0_df['poses_binary'] = section_0_df.apply(lambda row: row.subject_number_of_poses_0 > 10, axis=1)
    print(section_0_df.head)

    xls = pd.ExcelFile("data/study_summer_data.xlsx")
    general_data = xls.parse(0)
    print(general_data.columns)

    xls = pd.ExcelFile("data/BFI_scores.xlsx")
    bfi_df = xls.parse(0)

    all_data_df = pd.concat([general_data, section_0_df, section_1_df, section_2_df, bfi_df], axis=1)
    print(all_data_df.head)

    relevant_df = all_data_df[['average_grades', 'psychometric_grade',
                               'Openness', 'Conscientiousness', 'Extraversion',
                               'subject_number_of_poses_0', 'poses_binary',
                               'min_matrix_error_1',
                               'm_number_of_poses',
                               'm_min_matrix_error']].dropna()
    print(relevant_df.head)

    result = sm.ols(formula="Extraversion ~ poses_binary",
                    data=relevant_df).fit()
    # result = sm.ols(formula="GPA ~ subject_number_of_poses", data=relevant_df).fit()
    print result.summary()



analysis_3()

