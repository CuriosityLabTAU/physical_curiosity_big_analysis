#imports:
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from factor_analyzer import FactorAnalyzer
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 ,f_regression, mutual_info_regression
from copy import *


# load all the relevant data
# raw_data = pickle.load(open('data/raw_data_all_merged', 'r'))
matrix_error_data = pickle.load(open('data/matrix_error_data', 'r'))
optimal_user_error_sequence= pickle.load(open('data/optimal_user_error_sequence', 'r'))
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'rb'))
tasks_error_real_matrix = pickle.load(open('data/tasks_error_real_matrix', 'rb'))
tasks_error_subject_matrix = pickle.load(open('data/tasks_error_subject_matrix', 'rb'))
gamma_optimal_user_error_df = pickle.load(open('data/optimal_user_error_sequence', 'r'))
# gamma_optimal_user_error_df = pd.read_csv('data/gamma_optimal_user_error_df_goren.csv')


# === data processing ===

# figure 1: how did we get poses
#--- this is done in the file: graph_for_poses.py ---

# figure 2: definitions of measures
# axis: x-axis p, y-axis matrix error
# draw for one subject for section two
# draw matrix error, and optimal error
# in PPT: mark b_global (min), b_local(area), b_sequence (area between)
def figure_2():
    step_n=2
    subject_id_n=8

    # collect matrix error data:
    matrix_error=[]
    for subject_id, step in matrix_error_data.items():
        if subject_id==subject_id_n:
            for step_id, errors in step.items():
                if step_id == step_n:
                    if 'error' in errors.keys():
                        if len(errors['error']) > 0:
                            matrix_error = errors['error']
                            break

    # collect matrix optimal error:
    optimal_error=optimal_user_error_sequence[subject_id_n][step_n]
    x=[i for i in range(len(matrix_error))]

    #plot
    # plt.figure()
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.plot(x, matrix_error ,label='Matrix error')
    plt.plot(x, optimal_error,label='Optimal error')

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error and optimal error - for one subject for section two')


    plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=5)

    plt.show()


# figure 3: what is matrix error
# axis: x-axis p, y-axis matrix error
# draw for all subjects (not average) for section 2
def figure_3():
    step_n=2

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    for subject_id, step in matrix_error_data.items():
            for step_id, errors in step.items():
                if step_id == step_n:
                    if 'error' in errors.keys():
                        if len(errors['error']) > 0:
                            y=errors['error']
                            x = [i for i in range(len(y))]

                            plt.plot(x, y)

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error and optimal error - for all subjects for section two')
    plt.ylim([0, 2])

    plt.show()

# figure 4: what is matrix error
# axis: x-axis p, y-axis matrix error
# draw average over participants, for all sections
def figure_4():

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # data collection
    steps=[0,2,4,5,7,8,10,11]
    step_dict={0:[], 2:[], 4:[], 5:[], 7:[], 8:[], 10:[], 11:[]}
    for subject_id, step in matrix_error_data.items():
            for step_id, errors in step.items():
                if step_id==12:
                    continue
                if 'error' in errors.keys():
                    if len(errors['error']) > 0:
                        step_dict[step_id].append(errors['error'])
    n = 1
    for step in steps:

        data=step_dict[step]
        length = len(sorted(data, key=len, reverse=True)[0])
        data_array = np.array([xi + [np.nan] * (length - len(xi)) for xi in data])

        y = np.nanmean(data_array, axis=0)
        x=[i for i in range(y.shape[0])]

        plt.plot(x, y,label='sections %d'%n)

        n+=1

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error - average over participants, for all sections')
    plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=5)
    plt.ylim([0, 1.5])
    plt.xlim([0, 15])
    plt.show()

# figure 5: how did we compute slopes
# axis: x-axis sessions 2-8, y-axis: for each measure X6
# draw for one subject (highest R^2), points and slope
# TODO: choose the subject with the minimal sum_R (minimum because R < 0
def linear_regression_plot(data,_xlabel,_ylabel,_title):

    # start from 0:
    y = data - data[0]
    len_x = len(y)
    y = np.array(y)
    x = [i for i in range(len_x)]

    # crate x for a non intercepted linear regressio
    x = np.array(x)
    x = x[:, np.newaxis]

    # run linear regression
    m, _, _, _ = np.linalg.lstsq(x, y)

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.plot(x, y, '.')
    plt.plot(x, m * x , '-')

    plt.ylabel(_ylabel)
    plt.xlabel(_xlabel)
    plt.title(_title)
    plt.show()

def linear_regression_from_df(data,m_name,r_name):
    subjects=[]
    m_list=[]
    r_list=[]
    for row in data.iterrows():

        y = row[1].values.tolist()
        len_x = len(y)
        y = np.array(y)
        x = [i for i in range(len_x)]
        x = np.array(x)

        #take out nones:
        none_index=np.argwhere(np.isnan(y))
        y=np.delete(y, none_index, 0)
        x=np.delete(x, none_index, 0)
        #start from 0:
        y=y-y[0]

        # crate x for a non intercepted linear regressio
        x = x[:, np.newaxis]

        #run linear regression
        m, _, _, _ = np.linalg.lstsq(x, y)
        m_list.append(m)
        subjects.append(row[0])

        model, resid = np.linalg.lstsq(x, y)[:2]

        r2 = 1 - resid / (y.size * y.var())
        r_list.append(r2)

    df = pd.DataFrame(np.column_stack([m_list,r_list]),columns=[m_name,r_name],index=subjects)
    df.columns.names = ['subject_id']
    return df

def figure_5():
    ## --Number of poses--
    subject_number_of_poses

    # crate df:
    subject_number_of_poses_df = pd.DataFrame.from_dict(subject_number_of_poses, orient='index')
    subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1,
                                    inplace=True)  # delete the second epoch(no learning)
    other_sections_data = pd.DataFrame(subject_number_of_poses_df.iloc[:, 2:8])
    linear_regression_plot(other_sections_data.values[2],'section','number of poses','linear regression of number of poses in each section')
    r_m_number_of_poses_df=linear_regression_from_df(other_sections_data,'m_subject_number_of_poses','r_subject_number_of_poses')

    ##--Matrix error:--
    # crate df:
    min_dict = {}
    sum_dict = {}
    for subject_id, step in matrix_error_data.items():
        min_dict[subject_id] = {}
        sum_dict[subject_id] = {}

        for step_id, errors in step.items():
            if 'error' in errors.keys():
                if len(errors['error']) > 0:
                    min_error = min(errors['error'])
                    sum_error = np.nanmean(errors['error'])
                    min_dict[subject_id][step_id] = min_error
                    sum_dict[subject_id][step_id] = sum_error
                else:
                    min_dict[subject_id][step_id] = 1
                    sum_dict[subject_id][step_id] = 100
            else:
                min_dict[subject_id][step_id] = 1
                sum_dict[subject_id][step_id] = 100

    min_matrix_error = pd.DataFrame.from_dict(min_dict, orient='index')
    min_matrix_error.columns.names = ['subject_id']

    sum_matrix_error = pd.DataFrame.from_dict(sum_dict, orient='index')
    sum_matrix_error.columns.names = ['subject_id']

    #all other sections:
    other_sections_min_matrix_data= pd.DataFrame(min_matrix_error.iloc[:,2:8])
    linear_regression_plot(other_sections_min_matrix_data.values[46],'section','min matrix error','linear regression of min matrix error in each section')
    r_m_min_matrix_df=linear_regression_from_df(other_sections_min_matrix_data,'m_min_matrix','r_min_matrix')


    other_sections_sum_matrix_data= pd.DataFrame(sum_matrix_error.iloc[:,2:8])
    linear_regression_plot(other_sections_sum_matrix_data.values[46],'section','sum matrix error','linear regression of sum matrix error in each section')
    r_m_sum_matrix_df = linear_regression_from_df(other_sections_sum_matrix_data,'m_sum_matrix','r_sum_matrix')


    ##--Task error - real matrix--:
    # crate df:
    pass_threshold = 20

    task_error_real_matrix_results = {}
    for subject_id, step in tasks_error_real_matrix.items():

        task_error_real_matrix_results[subject_id] = {}

        for step_id, step in step.items():

            task_error_real_matrix_results[subject_id][step_id] = 0

            step_results = []

            for section_id in step.keys():

                if 'min_error' in tasks_error_real_matrix[subject_id][step_id][section_id].keys():
                    step_results.append(tasks_error_real_matrix[subject_id][step_id][section_id]['min_error'])

            if len(step_results) > 0:
                task_error_real_matrix_results[subject_id][step_id] = np.nanmean(step_results)

            else:
                task_error_real_matrix_results[subject_id][step_id] = 360

    task_error_real_matrix_results_df = pd.DataFrame.from_dict(task_error_real_matrix_results, orient='index')
    task_error_real_matrix_results_df.drop(task_error_real_matrix_results_df.columns[[0, 9]], axis=1, inplace=True)


    # all other sections:
    other_sections_task_error_real_matrix_results_data = pd.DataFrame(task_error_real_matrix_results_df.iloc[:, 2:])
    linear_regression_plot(other_sections_task_error_real_matrix_results_data.values[54],'section','task error','linear regression of task error real matrix in each section')
    r_m_task_error_real_matrix_df=linear_regression_from_df(other_sections_task_error_real_matrix_results_data,'m_task_error_real_matrix','r_task_error_real_matrix')



    ##--Task error - subject matrix--:

    #crate df:
    pass_threshold=20

    task_error_subject_matrix_results={}
    for subject_id, step in tasks_error_subject_matrix.items():

        task_error_subject_matrix_results[subject_id] = {}

        for step_id, step in step.items():

            task_error_subject_matrix_results[subject_id][step_id] = 0

            step_results=[]

            if subject_id==15.0:
                pass

            if step is not None:
                for section_id in step.keys():

                    if 'min_error' in tasks_error_subject_matrix[subject_id][step_id][section_id].keys():

                        step_results.append(tasks_error_subject_matrix[subject_id][step_id][section_id]['min_error'])



            if len(step_results)>0:
                task_error_subject_matrix_results[subject_id][step_id]=np.nanmean(step_results)

            else:
                task_error_subject_matrix_results[subject_id][step_id]=360

    task_error_subject_matrix_results_df=pd.DataFrame.from_dict(task_error_subject_matrix_results, orient='index')
    task_error_subject_matrix_results_df.drop(task_error_subject_matrix_results_df.columns[[8]], axis=1, inplace=True)

    #all other sections:
    other_sections_task_error_subject_matrix_results_data= pd.DataFrame(task_error_subject_matrix_results_df.iloc[:,2:])
    linear_regression_plot(other_sections_task_error_subject_matrix_results_data.values[20],'section','task error','linear regression of task error subject matrix in each section')
    r_m_task_error_subject_matrix_df = linear_regression_from_df(other_sections_task_error_subject_matrix_results_data,'m_task_error_subject_matrix','r_task_error_subject_matrix')


    ##-- Behavior gamma--

    #all other sections df
    other_sections_gamma= pd.DataFrame(gamma_optimal_user_error_df.iloc[:,2:8])
    linear_regression_plot(other_sections_gamma.values[46],'section','behavior gamma','linear regression of behavior gamma in each section')
    r_m_gamma_df=linear_regression_from_df(other_sections_gamma,'m_gamma','r_gamma')

    m_df=pd.concat([r_m_number_of_poses_df['m_subject_number_of_poses'],r_m_min_matrix_df['m_min_matrix'],
                    r_m_sum_matrix_df['m_sum_matrix'], r_m_task_error_real_matrix_df['m_task_error_real_matrix'],
                    r_m_task_error_subject_matrix_df['m_task_error_subject_matrix'],r_m_gamma_df['m_gamma']], axis=1)

    r_df=pd.concat([r_m_number_of_poses_df['r_subject_number_of_poses'],r_m_min_matrix_df['r_min_matrix'],
                    r_m_sum_matrix_df['r_sum_matrix'], r_m_task_error_real_matrix_df['r_task_error_real_matrix'],
                    r_m_task_error_subject_matrix_df['r_task_error_subject_matrix'],r_m_gamma_df['r_gamma']], axis=1)

    r_df['min_r'] = r_df[['r_subject_number_of_poses', 'r_min_matrix', 'r_sum_matrix', 'r_task_error_real_matrix',
                              'r_task_error_subject_matrix','r_gamma']].min(axis=1)
    subject_max_min_r=r_df['min_r'].idxmax(axis=0)

    # for i in
    #     sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    #
    #     plt.plot(x, y, '.')
    #     plt.plot(x, m * x, '-')
    #
    #     plt.ylabel(_ylabel)
    #     plt.xlabel(_xlabel)
    #     plt.title(_title)
    #     plt.show()
    #todo: df with all and min
    #todo: max on min
    #todo:plot the 5 linar

# figure_5()




# === data analysis ====
measures_data = {}
measures_data['section_1'] = pd.read_excel('data/big_analysis_normalized.xlsx', sheetname='section_0')
measures_data['section_2'] = pd.read_excel('data/big_analysis_normalized.xlsx', sheetname='section_1')
measures_data['section_9'] = pd.read_excel('data/big_analysis_normalized.xlsx', sheetname='section_9')
measures_data['section_other'] = pd.read_excel('data/big_analysis_normalized.xlsx', sheetname='other_sections')

# remove outliers
# in section 1, only those who did something
# measures_data['section_1'] = measures_data['section_1'][measures_data['section_1']['subject_number_of_poses_0'] > 0]
# error cannot be too big
# measures_data['section_1'] = measures_data['section_1'][measures_data['section_1']['sum_matrix_error_0'] < 100]
# min matrix error cannot be too big
# measures_data['section_1'] = measures_data['section_1'][measures_data['section_1']['min_matrix_error_0'] < 1]
# in section 2, only with more than 3
# measures_data['section_2'] = measures_data['section_2'][measures_data['section_2']['subject_number_of_poses_1'] > 3]
measures_data['section_1'] = measures_data['section_1'][measures_data['section_1']['behavior_gamma_0'] < 100]
measures_data['section_1'] = measures_data['section_1'][measures_data['section_1']['min_matrix_error_0'] < 1]
measures_data['section_2'] = measures_data['section_2'][measures_data['section_2']['behavior_gamma_1'] < 100]





## external data:
external_AQ = pd.read_excel("data/AQ_scores.xlsx")
external_BFI = pd.read_excel("data/BFI_scores.xlsx")
external_general=pd.read_excel("data/study_summer_data.xlsx")
external_tablet=pd.read_excel("data/data_from_tablet.xlsx")
#fix index for tablet data:
subject_id_for_tablet=external_tablet['subject_id']
external_tablet.index=subject_id_for_tablet

# all_external_data=pd.concat([external_BFI['Openness'], external_BFI['Neuroticism'],
#                              external_general[['age', 'gender', 'average_grades', 'psychometric_grade']], external_tablet['CEI_II_Total']], axis=1)
#

all_external_data = pd.concat([external_AQ['AQ_total_score'], external_BFI['Openness'], external_BFI['Neuroticism'],
                               external_general[['age', 'gender', 'average_grades', 'psychometric_grade']],
                               external_tablet['CEI_II_Total']], axis=1)

all_measures = pd.concat([measures_data['section_1'],
                          measures_data['section_2'], measures_data['section_other'], measures_data['section_9']], axis=1)

all_data_df=pd.concat([all_measures, all_external_data,], axis=1)



# figure 6: measures' histograms
# 6 X 4 subplots
# each row is a section (1, 2, overline, 9)
# each column is a measure
# each subplot is a histogram of that measure
def figure_6():
    # subplot
    for section in ['section_1', 'section_2', 'section_other', 'section_9']:
        sec_data = measures_data[section]
        for col in sec_data.columns:
            plt.hist(sec_data[col].values)
            plt.title('%s %s' %(section, col))
            plt.show()


# figure_6()

# figure 7: measures' correlation
# matrix/table: 24 X 24 (all measure X all measures
# Value is R, but only if p<0.05
def figure_7():
    corr_coef = all_measures.corr()
    plt.matshow(all_measures.corr())
    plt.colorbar()
    # plt.show()

    print(corr_coef)

# figure_7()

# === internal insights ===
# figure 8: do participants who explore more learn more with time
# formula: (\delta^i | \tilde\delta) ~ matrix^i + i + (n_pose^i | b_...)

step_data = pd.read_excel('data/all_data_normalized.xlsx')
# step_data = step_data[step_data['task_error_real_matrix'] < 100]
# step_data = step_data[step_data['min_matrix_error'] < 1]
# step_data = step_data[step_data['sum_matrix_error'] < 100]


def figure_8():
    result = sm.ols(formula="task_error_real_matrix ~ step_id + C(matrix) + number_of_poses -1",
                    data=step_data).fit()
    print result.summary()
    # conclusion: matrix effected, number of poses effect task error
    #             step_id is not correlated,
    # ==> they did not reduce the task error with steps (time)

    # result = sm.ols(formula="task_error_subject_matrix ~ step_id + C(matrix) + number_of_poses -1",
    #                 data=step_data).fit()
    # print result.summary()


    result = sm.ols(formula="min_matrix_error ~ step_id  + C(matrix) -1",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time

    result = sm.ols(formula="sum_matrix_error ~ step_id  + C(matrix) -1",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time

    result = sm.ols(formula="behavior_gamma ~ step_id  + C(matrix) -1",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time
    #    ==> participants improved their poses selection, but not order
    #    ==> learned "macro" exploration, but not "micro" exploration

    # =====> the task was too hard to learn
    # they did not improve in task performance
    # they did not improve in local/micro exploration
    # they did improve their exploration strategy

# figure_8()

# figure 8.a: x-axis: number of poses, y-axis:task_error_real_matrix
#       plot all data points, and the linear line that describes them
def figure_8a():
    result = sm.ols(formula="task_error_real_matrix ~ number_of_poses",
                    data=step_data).fit()
    print result.summary()
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.jointplot(x="number_of_poses", y="task_error_real_matrix", data=step_data, kind="reg")
    # sns.regplot(x="number_of_poses", y="task_error_real_matrix", data=step_data)
    plt.xlabel('Number of poses')
    plt.ylabel(r"$\delta$" +'- error in task')
    plt.title('Linear regression between number of poses and task error ' +r"($\delta$)")

    plt.show()


# figure 8.b: x-axis: step_id, y-ais: min_matrix_error
#       plot all data points, and the linear line that describes them
def figure_8b():
    result = sm.ols(formula="min_matrix_error ~ step_id",
                    data=step_data).fit()
    print result.summary()
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.jointplot(x="step_id", y="min_matrix_error", data=step_data, kind="reg")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    # sns.regplot(x="step_id", y="min_matrix_error", data=step_data)
    plt.xlabel('Section id')
    plt.ylabel(r"$b_{global}$ -"+'min matrix error')
    plt.title('Linear regression between section id and min matrix error '+r"($b_{global})$")

    plt.show()

# === external validation ===


# formula '(CEI | BFI | PET) ~ factor(measures)'

# ad_hoc (figure 7) ==> 3 measures



def figure_9():

    fa = FactorAnalyzer()
    fa.analyze(all_measures, len(all_measures.columns), rotation='promax')
    ev, v = fa.get_eigenvalues()
    plt.plot(ev)
    # plt.show()

    n_factors = 4
    fa.analyze(all_measures, n_factors, rotation='promax')
    for i in range(1, 1 + n_factors):
        factor_name = 'Factor%d' % i
        print('----- %s ------' % factor_name)
        print(fa.loadings[factor_name][abs(fa.loadings[factor_name]) > 0.4])

    x = np.zeros([all_measures.shape[0], n_factors])
    for i in range(1, 1 + n_factors):
        factor_name = 'Factor%d' % i
        # print(np.expand_dims(fa.loadings[factor_name], 1).shape)
        # print(all_measures.values.shape)
        relevant_factors = abs(fa.loadings[factor_name]) > 0.4
        weighted_measures = fa.loadings[factor_name][relevant_factors] / np.sum(abs(fa.loadings[factor_name][relevant_factors]))
        relevant_measures = all_measures.values[:, relevant_factors.values]
        x[:, i - 1] = np.squeeze(np.dot(relevant_measures, np.expand_dims(weighted_measures, 1)))
        # x[:, i-1] = np.squeeze(np.dot(all_measures.values,  np.expand_dims(fa.loadings[factor_name], 1)))


    # convert x into data_frame, columns = factor_1, factor_2
    factor_names = ['factor_%d' % s for s in range(n_factors)]
    factor_df = pd.DataFrame(x, columns=factor_names,index=all_measures.index)
    # print factor_df

    # factor df and external_data df:
    factors_and_external_df = pd.concat([factor_df, all_external_data], axis=1)

    interesting_measures = ['psychometric_grade', 'AQ_total_score', 'Openness', 'Neuroticism',
                            'CEI_II_Total','average_grades']

    for i_m in interesting_measures:
        the_formula = i_m + ' ~ '
        for fn in factor_names:
            the_formula += fn + ' +'
        the_formula = the_formula[:-2]
        print(the_formula)
        result = sm.ols(formula=the_formula, data=factors_and_external_df).fit()
        print result.summary()

def figure_10():
    interesting_measures = ['psychometric_grade', 'AQ_total_score', 'Openness', 'Neuroticism',
                            'CEI_II_Total','average_grades']
    measures=['m_behavior_gamma','m_task_error_real_matrix_results']

    for i_m in interesting_measures:
        the_formula = i_m + ' ~ '
        for measure in measures:
            the_formula += measure + ' +'
        the_formula = the_formula[:-2]
        print(the_formula)
        result = sm.ols(formula=the_formula, data=all_data_df).fit()
        print result.summary()


# figure_10()

all_measures = all_measures.dropna().apply(zscore)
# print(all_measures)
# figure_9()



def figure_11():
    # hist of matrix and measures
    measures = ['number_of_poses', 'min_matrix_error', 'sum_matrix_error', 'task_error_real_matrix',
                'task_error_subject_matrix', 'behavior_gamma']
    matrices = [0, 1, 2, 3, 4, 5, 6, 7]

    # subplot
    for measure in measures:
        sec_data = step_data[[measure,'matrix']]

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        i=1
        for mat in matrices:
            ax = fig.add_subplot(3, 3, i)
            sec_data.loc[sec_data['matrix'] == mat]
            ax.hist(sec_data.loc[sec_data['matrix'] == mat][measure].values)
            i+=1
            ax.set_title('%s' %(mat))

        fig.suptitle(measure)
        plt.show()

def all_subsets(xs):
  if not xs:
    return [[]]
  else:
    x = xs.pop()
    subsets = all_subsets(xs)
    subsets_copy = deepcopy(subsets) # NB you need to use a deep copy here!
    for s in subsets_copy:
      s.append(x)
    subsets.extend(subsets_copy)
    return subsets

def figure_12():
    interesting_measures = ['psychometric_grade', 'AQ_total_score', 'Openness', 'Neuroticism',
                            'CEI_II_Total','average_grades']

    measure_data = list(all_data_df)[:-8]
    while "subject_id" in measure_data: measure_data.remove('subject_id')

    results_dic={}
    sets=all_subsets([i for i in range(len(measure_data))])

    counter = len(sets)
    for i_m in interesting_measures:
        results_dic[i_m]=[]
        print "~~~~~~~~~~~~~~~~~~~"+i_m+"~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        for set in sets:
            if len(set) > 0:
                if len(set)<6:
                    measures= [measure_data[i] for i in set]

                    the_formula = i_m + ' ~ '
                    for measure in measures:
                        the_formula += measure + ' +'
                    the_formula = the_formula[:-2]

                    result = sm.ols(formula=the_formula, data=all_data_df).fit()

                    if result.f_pvalue <0.05:
                        pvalue=result.f_pvalue
                        rsquared=result.rsquared
                        results_dic[i_m].append(([pvalue, rsquared,measures]))

                        print "last p= ",result.f_pvalue,"last r= ",rsquared

    writer = pd.ExcelWriter('data/OLS_fit.xlsx', engine='xlsxwriter')

    for i_m in interesting_measures:

        i_m_df=pd.DataFrame(results_dic[i_m],columns=['pvalue','rsquared','measures'])

        i_m_df.to_excel(writer, sheet_name='%s' %i_m )

        # Close the Pandas Excel writer and output the Excel file.
    writer.save()

# figure_12()


def figure_13():
    result = sm.ols(formula="psychometric_grade ~ m_behavior_gamma",
                    data=all_data_df).fit()
    print result.summary()


def figure_14():
    result = sm.ols(formula="task_error_real_matrix ~ step_id + number_of_poses ",
                    data=step_data).fit()
    print result.summary()
    # conclusion: matrix effected, number of poses effect task error
    #             step_id is not correlated,
    # ==> they did not reduce the task error with steps (time)

    # result = sm.ols(formula="task_error_subject_matrix ~ step_id + C(matrix) + number_of_poses -1",
    #                 data=step_data).fit()
    # print result.summary()


    result = sm.ols(formula="min_matrix_error ~ step_id  ",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time

    result = sm.ols(formula="sum_matrix_error ~ step_id ",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time

    result = sm.ols(formula="behavior_gamma ~ step_id  ",
                    data=step_data).fit()
    print result.summary()
    # conclusion: subjects improved their exploration with time
    #    ==> participants improved their poses selection, but not order
    #    ==> learned "macro" exploration, but not "micro" exploration

    # =====> the task was too hard to learn
    # they did not improve in task performance
    # they did not improve in local/micro exploration
    # they did improve their exploration strategy

#task analysis
def figure_15():
    #data for analysis:
    step_data = pd.read_excel('data/all_data_normalized.xlsx')
    new_df = pd.DataFrame(columns=['step_id', 'matrix', 'task_number', 'task_performance'])

    for i in range(1, 17):
        temp_df = step_data.loc[step_data[i].dropna().index][['step_id', 'matrix', i]]
        temp_df = pd.melt(temp_df, id_vars=['step_id', 'matrix'], value_vars=i, value_name='task_performance')
        temp_df.columns = ['step_id', 'matrix', 'task_number', 'task_performance']
        new_df = new_df.append(temp_df)
    new_df = new_df.astype("float")

    #fit
    result = sm.ols(formula="task_performance ~ step_id ",
                    data=new_df).fit()
    print result.summary()

# =============== ROMAN =================
def print_result(result):
    print(result.summary())
    print('F(%d, %d)=%2.1f, p=%2.4f, R=%2.2f, \\beta=%2.5f' %
          (result.df_model, result.df_resid, result.fvalue, result.f_pvalue,
           np.sqrt(result.rsquared) * np.sign(result.params.values[1]),
           result.params.values[1]))

step_data = step_data[step_data['step_id'] > 0]
step_data = step_data[step_data['step_id'] < 8]
def roman_figure_1():
    the_measures = ['number_of_poses', 'min_matrix_error', 'sum_matrix_error', 'behavior_gamma',
                    'task_error_real_matrix', 'task_error_subject_matrix']
    the_formula = "%s ~ %s " % ('min_matrix_error', 'number_of_poses')
    result = sm.ols(formula=the_formula, data=step_data).fit()
    print_result(result)

    the_formula = "%s ~ %s " % ('sum_matrix_error', 'number_of_poses')
    result = sm.ols(formula=the_formula, data=step_data).fit()
    print_result(result)

    the_formula = "%s ~ %s " % ('behavior_gamma', 'number_of_poses')
    result = sm.ols(formula=the_formula, data=step_data).fit()
    print_result(result)

    the_formula = "%s ~ %s " % ('task_error_real_matrix', 'min_matrix_error')
    result = sm.ols(formula=the_formula, data=step_data).fit()
    print_result(result)

    the_formula = "%s ~ %s " % ('task_error_subject_matrix', 'min_matrix_error')
    result = sm.ols(formula=the_formula, data=step_data).fit()
    print_result(result)

def roman_figure_2():
    the_measures = ['number_of_poses', 'min_matrix_error', 'sum_matrix_error', 'behavior_gamma',
                    'task_error_real_matrix', 'task_error_subject_matrix']
    for the_measure in the_measures:
        the_formula = "%s ~ step_id + C(matrix) - 1" % the_measure
        result = sm.ols(formula=the_formula, data=step_data).fit()
        print(result.summary())

# roman_figure_1()