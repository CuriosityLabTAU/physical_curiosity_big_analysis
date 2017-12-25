###Imports:
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from external_files.angle_matrix import AngleMatrix
from numpy.linalg import inv,pinv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

##we will crate a DF with all the data:

## Number of poses
#lode data
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'rb'))

#crate df:
subject_number_of_poses_df=pd.DataFrame.from_dict(subject_number_of_poses, orient='index')
subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1, inplace=True)  #delete the second epoch(no learning)
subject_number_of_poses_df.reset_index(inplace=True)
subject_number_of_poses_df.columns = ['subject_id']+[i for i in range(9)]
keys = [i for i in range(9)]

three_columns_df=pd.melt(subject_number_of_poses_df,id_vars='subject_id', value_vars=keys, value_name='number_of _poses')
three_columns_df.columns=['subject_id','step_id','number_of_poses']


##Matrix error:
# lode data
matrix_error = pickle.load(open('data/matrix_error_data', 'rb'))

#crate df:
min_dict={}
mean_dict={}
for subject_id,step in matrix_error.items():
    min_dict[subject_id] = {}
    mean_dict[subject_id] = {}

    for step_id, errors in step.items():
        if 'error' in errors.keys():
            if len(errors['error'])>0:
                min_error=min(errors['error'])
                mean_error=np.nanmean(errors['error'])
                min_dict[subject_id][step_id] = min_error
                mean_dict[subject_id][step_id] = mean_error
            else:
                min_dict[subject_id][step_id] = None
                mean_dict[subject_id][step_id] = None
        else:
            min_dict[subject_id][step_id] = None
            mean_dict[subject_id][step_id] = None

#min
min_matrix_error=pd.DataFrame.from_dict(min_dict, orient='index')
min_matrix_error.columns.names=['subject_id']

min_matrix_error.reset_index(inplace=True)
min_matrix_error.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
min_matrix_error_three_columns_df=pd.melt(min_matrix_error,id_vars='subject_id', value_vars=keys, value_name='min_matrix_error')
min_matrix_error_three_columns_df.columns=['subject_id','step_id','min_matrix_error']

#mean
mean_matrix_error=pd.DataFrame.from_dict(mean_dict, orient='index')
mean_matrix_error.columns.names=['subject_id']

mean_matrix_error.reset_index(inplace=True)
mean_matrix_error.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
mean_matrix_error_three_columns_df=pd.melt(mean_matrix_error,id_vars='subject_id', value_vars=keys, value_name='mean_matrix_error')
mean_matrix_error_three_columns_df.columns=['subject_id','step_id','mean_matrix_error']


#Join
three_columns_df = pd.merge(three_columns_df, min_matrix_error_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])
three_columns_df = pd.merge(three_columns_df, mean_matrix_error_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])


##Task error - real matrix:
#lode data
tasks_error_real_matrix = pickle.load(open('data/tasks_error_real_matrix', 'rb'))

#crate df:
pass_threshold=20

task_error_real_matrix_results={}
for subject_id, step in tasks_error_real_matrix.items():

    task_error_real_matrix_results[subject_id] = {}

    for step_id, step in step.items():

        task_error_real_matrix_results[subject_id][step_id] = 0

        step_results=[]

        for section_id in step.keys():

            if 'min_error' in tasks_error_real_matrix[subject_id][step_id][section_id].keys():

                step_results.append(tasks_error_real_matrix[subject_id][step_id][section_id]['min_error'])



        if len(step_results)>0:
            task_error_real_matrix_results[subject_id][step_id]=np.nanmean(step_results)

        else:
            task_error_real_matrix_results[subject_id][step_id]=None

task_error_real_matrix_results_df=pd.DataFrame.from_dict(task_error_real_matrix_results, orient='index')
task_error_real_matrix_results_df.drop(task_error_real_matrix_results_df.columns[[0]], axis=1, inplace=True)

task_error_real_matrix_results_df.reset_index(inplace=True)
task_error_real_matrix_results_df.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
task_error_real_matrix_three_columns_df=pd.melt(task_error_real_matrix_results_df,id_vars='subject_id', value_vars=keys, value_name='task_error_real_matrix')
task_error_real_matrix_three_columns_df.columns=['subject_id','step_id','task_error_real_matrix']

#Join
three_columns_df = pd.merge(three_columns_df, task_error_real_matrix_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])



##Task error - subject matrix:
#lode data
tasks_error_subject_matrix = pickle.load(open('data/tasks_error_subject_matrix', 'rb'))

#crate df:
pass_threshold=20

task_error_subject_matrix_results={}
for subject_id, step in tasks_error_subject_matrix.items():

    task_error_subject_matrix_results[subject_id] = {}

    for step_id, step in step.items():

        task_error_subject_matrix_results[subject_id][step_id] = 0

        step_results=[]

        if step is not None:
            for section_id in step.keys():

                if 'min_error' in tasks_error_subject_matrix[subject_id][step_id][section_id].keys():

                    step_results.append(tasks_error_subject_matrix[subject_id][step_id][section_id]['min_error'])



        if len(step_results)>0:
            task_error_subject_matrix_results[subject_id][step_id]=np.nanmean(step_results)

        else:
            task_error_subject_matrix_results[subject_id][step_id]=None

task_error_subject_matrix_results_df=pd.DataFrame.from_dict(task_error_subject_matrix_results, orient='index')

task_error_subject_matrix_results_df.reset_index(inplace=True)
task_error_subject_matrix_results_df.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
task_error_subject_matrix_three_columns_df=pd.melt(task_error_subject_matrix_results_df,id_vars='subject_id', value_vars=keys, value_name='task_error_subject_matrix')
task_error_subject_matrix_three_columns_df.columns=['subject_id','step_id','task_error_subject_matrix']

#Join
three_columns_df = pd.merge(three_columns_df, task_error_subject_matrix_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])



##Matrix of step\subject:
poses = pickle.load(open('data/data_of_poses_21', 'rb'))

# createing matrix:
matrix_info = {}
for subject_id, step in poses.items():

    matrix_info[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['transformation']

        matrix_info[subject_id][step_id]= matrix

# crate df:
matrix_info_df = pd.DataFrame.from_dict(matrix_info, orient='index')
matrix_info_df.drop(matrix_info_df.columns[[1]], axis=1,inplace=True)  # delete the second epoch(no learning)


matrix_info_df.reset_index(inplace=True)
matrix_info_df.columns = ['subject_id'] + [i for i in range(9)]
keys = [i for i in range(9)]

matrix_info_df_three_columns_df = pd.melt(matrix_info_df, id_vars='subject_id', value_vars=keys,value_name='matrix')
matrix_info_df_three_columns_df.columns = ['subject_id', 'step_id', 'matrix']

#Join
three_columns_df = pd.merge(three_columns_df, matrix_info_df_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])

##export to excel
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('data/all_data.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
three_columns_df.to_excel(writer, sheet_name='all_data')


# Close the Pandas Excel writer and output the Excel file.
writer.save()