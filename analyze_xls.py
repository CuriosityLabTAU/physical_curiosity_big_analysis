#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer


base_path = 'C:/Goren/CuriosityLab/Code/python/physical_curiosity_big_analysis/data/'


def analysis_1():
    xls = pd.ExcelFile("%s/big_analysis.xlsx" % base_path)
    section_0 = xls.parse(0)
    section_1 = xls.parse(1)
    print(section_0.columns)

    xls = pd.ExcelFile("%s/study_summer_data.xlsx" % base_path)
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
    xls = pd.ExcelFile("%s/all_data.xlsx" % base_path)
    data_df = xls.parse(0)

    data_clean = data_df.drop(data_df[data_df.step_id == 8].index)
    data_clean = data_clean.drop(data_clean[data_clean.step_id == 0].index)

    result = sm.ols(formula="number_of_poses ~ step_id + C(matrix) -1", data=data_clean).fit()
    result = sm.ols(formula="number_of_poses ~ step_id + C(matrix) -1", data=data_clean).fit()

    print(result.summary())

def analysis_3():
    xls = pd.ExcelFile("%s/big_analysis.xlsx" % base_path)
    section_0_df = xls.parse(0)
    section_1_df = xls.parse(1)
    section_2_df = xls.parse(2)

    section_0_df['poses_binary'] = section_0_df.apply(lambda row: row.subject_number_of_poses_0 > 10, axis=1)
    print(section_0_df.head)

    xls = pd.ExcelFile("%s/study_summer_data.xlsx" % base_path)
    general_data = xls.parse(0)
    print(general_data.columns)

    xls = pd.ExcelFile("%s/BFI_scores.xlsx" % base_path)
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

def analysis_4():
    xls = pd.ExcelFile("%s/all_data.xlsx"  % base_path)
    data_df = xls.parse(0)

    data_clean = data_df.drop(data_df[data_df.step_id == 8].index)
    data_clean = data_clean.drop(data_clean[data_clean.step_id == 0].index)

    print data_clean
    df3 = data_clean[data_clean['task_error_subject_matrix'] < 100]

    ax = sns.boxplot(x="matrix", y="task_error_subject_matrix", data=df3)

    plt.show()

def unite_data():
    xls = pd.ExcelFile("%s/big_analysis.xlsx"  % base_path)
    section_0 = xls.parse(0)
    section_1 = xls.parse(1)
    section_2 = xls.parse(2)

    xls = pd.ExcelFile("%s/study_summer_data.xlsx"  % base_path)
    general_data = xls.parse(0)

    xls = pd.ExcelFile("%s/BFI_scores.xlsx"  % base_path)
    bfi_df = xls.parse(0)

    xls = pd.ExcelFile("%s/aq_scores.xlsx" % base_path)
    aq_df = xls.parse(0)

    all_data_df = pd.concat([general_data, section_0, section_1, section_2, bfi_df, aq_df], axis=1)
    print('all data:', len(all_data_df), len(all_data_df.columns))
#
#
    u'average_grades',


# u'psychometric_grade',
# u'שולט ברובוט',
# u'שאלה 1',
# u'שאלה 2',
# u'שאלה 3',

    relevant_data = all_data_df[[      #u'age',
            u'subject_number_of_poses_0',
            u'min_matrix_error_0',
        u'mean_matrix_error_0',
                                       u'task_error_real_matrix_results_0',
                                       u'task_error_subject_matrix_results_0',
                                       u'subject_number_of_poses_1',
                        u'min_matrix_error_1',
                       u'mean_matrix_error_1',
          u'task_error_real_matrix_results_1',
       u'task_error_subject_matrix_results_1',
                         u'm_number_of_poses',
                        u'm_min_matrix_error',
                       u'm_mean_matrix_error',
          u'm_task_error_real_matrix_results',
       u'm_task_error_subject_matrix_results'#,
                       #        u'Extraversion',
                       #       u'Agreeableness',
                       #   u'Conscientiousness',
                       #         u'Neuroticism',
                       #            u'Openness',
                       #        u'social_skill',
                       # u'attention_switching',
                       # u'attention_to_detail',
                       #       u'communication',
                       #         u'imagination',
                       #         u'total_score',
                       #          u'bfi_total_score'
                                       ]]

    return relevant_data

data_df = unite_data()
data_df = data_df.dropna()
print(len(data_df), len(data_df.columns))


fa = FactorAnalyzer()
fa.analyze(data_df, len(data_df.columns), rotation='promax')
ev, v = fa.get_eigenvalues()
plt.plot(ev)
plt.show()

n_factors = 4
fa.analyze(data_df, n_factors, rotation='promax')
for i in range(1, 1 + n_factors):
    factor_name = 'Factor%d' % i
    print('----- %s ------' % factor_name)
    print(fa.loadings[factor_name][abs(fa.loadings[factor_name]) > 0.4])
