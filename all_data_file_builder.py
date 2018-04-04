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
sum_dict={}
for subject_id,step in matrix_error.items():
    min_dict[subject_id] = {}
    sum_dict[subject_id] = {}

    for step_id, errors in step.items():
        if 'error' in errors.keys():
            if len(errors['error'])>0:
                min_error=min(errors['error'])
                sum_error=np.nanmean(errors['error'])
                min_dict[subject_id][step_id] = min_error
                sum_dict[subject_id][step_id] = sum_error
            else:
                min_dict[subject_id][step_id] = 1
                sum_dict[subject_id][step_id] = 100
        else:
            min_dict[subject_id][step_id] = 1
            sum_dict[subject_id][step_id] = 100

#min
min_matrix_error=pd.DataFrame.from_dict(min_dict, orient='index')
min_matrix_error.columns.names=['subject_id']

min_matrix_error.reset_index(inplace=True)
min_matrix_error.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
min_matrix_error_three_columns_df=pd.melt(min_matrix_error,id_vars='subject_id', value_vars=keys, value_name='min_matrix_error')
min_matrix_error_three_columns_df.columns=['subject_id','step_id','min_matrix_error']

#mean
sum_matrix_error=pd.DataFrame.from_dict(sum_dict, orient='index')
sum_matrix_error.columns.names=['subject_id']

sum_matrix_error.reset_index(inplace=True)
sum_matrix_error.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
sum_matrix_error_three_columns_df=pd.melt(sum_matrix_error,id_vars='subject_id', value_vars=keys, value_name='sum_matrix_error')
sum_matrix_error_three_columns_df.columns=['subject_id','step_id','sum_matrix_error']


#Join
three_columns_df = pd.merge(three_columns_df, min_matrix_error_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])
three_columns_df = pd.merge(three_columns_df, sum_matrix_error_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])


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
####
## Behavior gamma
#lode data
behavior_gamma_df = pickle.load(open('data/gamma_user_vs_optimal_user', 'r'))

#crate df:
# behavior_gamma_df=pd.DataFrame.from_dict(gamma_optimal_user_error_df, orient='index')
# behavior_gamma_df.drop(behavior_gamma_df.columns[[1]], axis=1, inplace=True)  #delete the second epoch(no learning)
behavior_gamma_df.reset_index(inplace=True)
behavior_gamma_df.columns = ['subject_id']+[i for i in range(9)]
keys = [i for i in range(9)]

behavior_gamma_three_columns_df=pd.melt(behavior_gamma_df,id_vars='subject_id', value_vars=keys, value_name='behavior_gamma')
behavior_gamma_three_columns_df.columns=['subject_id','step_id','behavior_gamma']
#Join
three_columns_df = pd.merge(three_columns_df, behavior_gamma_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])

########################
##Task error - real matrix:
#lode data
tasks_error_real_matrix = pickle.load(open('data/tasks_error_real_matrix', 'rb'))

#task dict:
tasks={'two_hands_forward':1, 'two_hands_down':2, 'two_hands_to_the_side':3, 'two_hands_up':4,
       'right_hand_up_left_hand_down':5, 'right_hand_up_left_hand_forward':6, 'right_hand_up_left_hand_to_the_side':7, 'right_hand_forward_left_hand_down':8,
       'right_hand_forward_left_hand_side':9, 'right_hand_to_the_side_left_hand_down':10, 'right_hand_to_the_side_left_hand_forward':11, 'right_hand_down_left_hand_to_the_side':12,
       'right_hand_down_left_hand_forward':13, 'left_hand_up_right_hand_down':14, 'left_hand_up_right_hand_forward':15, 'left_hand_up_right_hand_to_the_side':16
       }

#crate df:

task_error_real_matrix={}
for subject_id, step in tasks_error_real_matrix.items():

    task_error_real_matrix[subject_id] = {}

    for step_id, step in step.items():

        task_error_real_matrix[subject_id][step_id] = {1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7:None,
                                                       8:None, 9:None, 10:None, 11:None, 12:None, 13:None, 14:None, 15:None, 16:None}

        for section_id in step.keys():

            if 'min_error' in tasks_error_real_matrix[subject_id][step_id][section_id].keys():
                task= tasks_error_real_matrix[subject_id][step_id][section_id]['task']
                task_number=tasks[task]

                task_error_real_matrix[subject_id][step_id][task_number]=tasks_error_real_matrix[subject_id][step_id][section_id]['min_error']

task_real_matrix_df=pd.DataFrame.from_dict(task_error_real_matrix, orient='index')
task_real_matrix_df.drop(task_real_matrix_df.columns[[0]], axis=1, inplace=True)

task_real_matrix_df.reset_index(inplace=True)
task_real_matrix_df.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
task_real_matrix_three_columns_df=pd.melt(task_real_matrix_df,id_vars='subject_id', value_vars=keys, value_name='task_real_matrix')
task_real_matrix_three_columns_df.columns=['subject_id','step_id','task_real_matrix']

task_real_matrix_three_columns_df[[i for i in range(1,17)]] = task_real_matrix_three_columns_df.task_real_matrix.apply(pd.Series)[[i for i in range(1,17)]]

task_real_matrix_three_columns_df=task_real_matrix_three_columns_df.drop('task_real_matrix', axis=1)

#Join
three_columns_df = pd.merge(three_columns_df, task_real_matrix_three_columns_df,  how='left', left_on=['subject_id','step_id'], right_on = ['subject_id','step_id'])

#######################
########################
##Task error - real matrix:
#lode data
tasks_error_subject_matrix = pickle.load(open('data/tasks_error_subject_matrix', 'rb'))

#task dict:
tasks={'two_hands_forward':1, 'two_hands_down':2, 'two_hands_to_the_side':3, 'two_hands_up':4,
       'right_hand_up_left_hand_down':5, 'right_hand_up_left_hand_forward':6, 'right_hand_up_left_hand_to_the_side':7, 'right_hand_forward_left_hand_down':8,
       'right_hand_forward_left_hand_side':9, 'right_hand_to_the_side_left_hand_down':10, 'right_hand_to_the_side_left_hand_forward':11, 'right_hand_down_left_hand_to_the_side':12,
       'right_hand_down_left_hand_forward':13, 'left_hand_up_right_hand_down':14, 'left_hand_up_right_hand_forward':15, 'left_hand_up_right_hand_to_the_side':16
       }

#crate df:

task_error_subject_matrix={}
for subject_id, step in tasks_error_subject_matrix.items():

    task_error_subject_matrix[subject_id] = {}

    for step_id, step in step.items():

        task_error_subject_matrix[subject_id][step_id] = {1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7:None,
                                                       8:None, 9:None, 10:None, 11:None, 12:None, 13:None, 14:None, 15:None, 16:None}

        for section_id in step.keys():

            if 'min_error' in tasks_error_subject_matrix[subject_id][step_id][section_id].keys():
                task= tasks_error_subject_matrix[subject_id][step_id][section_id]['task']
                task_number=tasks[task]

                task_error_subject_matrix[subject_id][step_id][task_number]=tasks_error_subject_matrix[subject_id][step_id][section_id]['min_error']

task_subject_matrix_df=pd.DataFrame.from_dict(task_error_subject_matrix, orient='index')
# task_subject_matrix_df.drop(task_subject_matrix_df.columns[[0]], axis=1, inplace=True)

task_subject_matrix_df.reset_index(inplace=True)
task_subject_matrix_df.columns = ['subject_id']+[i for i in range(9)]

keys = [i for i in range(9)]
task_subject_matrix_three_columns_df=pd.melt(task_subject_matrix_df,id_vars='subject_id', value_vars=keys, value_name='task_real_matrix')
task_subject_matrix_three_columns_df.columns=['subject_id','step_id','task_subject_matrix']

task_subject_matrix_three_columns_df[[i for i in range(1,17)]] = task_subject_matrix_three_columns_df.task_subject_matrix.apply(pd.Series)[[i for i in range(1,17)]]

task_subject_matrix_three_columns_df=task_subject_matrix_three_columns_df.drop('task_subject_matrix', axis=1)
#######################



##Task error - real matrix:

#crate df:
task_error=three_columns_df[[i for i in range(1,17)]].mean(axis=1)
three_columns_df['task_error_real_matrix']=task_error

######################




##Task error - subject matrix:

#crate df:
task_subject_matrix=task_subject_matrix_three_columns_df[[i for i in range(1,17)]].mean(axis=1)
three_columns_df['task_error_subject_matrix']=task_subject_matrix

######################


##export to excel
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('data/all_data.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
three_columns_df.to_excel(writer, sheet_name='all_data')


# Close the Pandas Excel writer and output the Excel file.
writer.save()