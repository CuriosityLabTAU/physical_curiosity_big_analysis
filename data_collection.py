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
from sklearn import datasets, linear_model

# linear regression fun:
def linear_regression_from_df(data,m_name):
    subjects=[]
    m_list=[]
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

    df = pd.DataFrame(np.column_stack([m_list]),columns=[m_name],index=subjects)
    df.columns.names = ['subject_id']
    return df


##we will crate a DF with all the data:
#section 0 df
#section 1 df
#all other sections


## Number of poses
#lode data
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'r'))

#crate df:
subject_number_of_poses_df=pd.DataFrame.from_dict(subject_number_of_poses, orient='index')

subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1, inplace=True)  #delete the second epoch(no learning)

#section 0 df
section_0_df=pd.DataFrame(subject_number_of_poses_df[0])
section_0_df.columns.names=['subject_id']
section_0_df.columns=['subject_number_of_poses']

#section 1 df
section_1_df=pd.DataFrame(subject_number_of_poses_df[2])
section_1_df.columns.names=['subject_id']
section_1_df.columns=['subject_number_of_poses']

#all other sections df
other_sections_data= pd.DataFrame(subject_number_of_poses_df.iloc[:,2:])
other_sections_df=linear_regression_from_df(other_sections_data,'m_number_of_poses')


##Matrix error:
#lode data
matrix_error = pickle.load(open('data/matrix_error_data', 'r'))

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


min_matrix_error=pd.DataFrame.from_dict(min_dict, orient='index')
min_matrix_error.columns.names=['subject_id']

mean_matrix_error=pd.DataFrame.from_dict(mean_dict, orient='index')
mean_matrix_error.columns.names=['subject_id']



#section 0:
section_0_min_matrix_df=pd.DataFrame(min_matrix_error[0])
section_0_min_matrix_df.columns.names=['subject_id']
section_0_min_matrix_df.columns=['min_matrix_error']
section_0_mean_matrix_df=pd.DataFrame(mean_matrix_error[0])
section_0_mean_matrix_df.columns.names=['subject_id']
section_0_mean_matrix_df.columns=['mean_matrix_error']

#section 1:
section_1_min_matrix_df=pd.DataFrame(min_matrix_error[2])
section_1_min_matrix_df.columns.names=['subject_id']
section_1_min_matrix_df.columns=['min_matrix_error']
section_1_mean_matrix_df=pd.DataFrame(mean_matrix_error[2])
section_1_mean_matrix_df.columns.names=['subject_id']
section_1_mean_matrix_df.columns=['mean_matrix_error']

#all other sections:
other_sections_min_matrix_data= pd.DataFrame(min_matrix_error.iloc[:,2:])
other_sections_min_matrix_df=linear_regression_from_df(other_sections_min_matrix_data,'m_min_matrix_error')

other_sections_mean_matrix_data= pd.DataFrame(mean_matrix_error.iloc[:,2:])
other_sections_mean_matrix_df=linear_regression_from_df(other_sections_mean_matrix_data,'m_mean_matrix_error')

#conect to df:
section_0_df = pd.concat([section_0_df, section_0_min_matrix_df,section_0_mean_matrix_df], axis=1)
section_1_df = pd.concat([section_1_df, section_1_min_matrix_df,section_1_mean_matrix_df], axis=1)
other_sections_df = pd.concat([other_sections_df, other_sections_min_matrix_df,other_sections_mean_matrix_df], axis=1)


##Task error - real matrix:
#lode data
tasks_error_real_matrix = pickle.load(open('data/tasks_error_real_matrix', 'r'))

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
task_error_real_matrix_results_df.drop(task_error_real_matrix_results_df.columns[[0, 9]], axis=1, inplace=True)


#section 0:
section_0_task_error_real_matrix_results_df=pd.DataFrame(task_error_real_matrix_results_df[1])
section_0_task_error_real_matrix_results_df.columns.names=['subject_id']
section_0_task_error_real_matrix_results_df.columns=['task_error_real_matrix_results']

#section 1:
section_1_task_error_real_matrix_results_df=pd.DataFrame(task_error_real_matrix_results_df[2])
section_1_task_error_real_matrix_results_df.columns.names=['subject_id']
section_1_task_error_real_matrix_results_df.columns=['task_error_real_matrix_results']

#all other sections:
other_sections_task_error_real_matrix_results_data= pd.DataFrame(task_error_real_matrix_results_df.iloc[:,2:])
other_sections_task_error_real_matrix_results_df=linear_regression_from_df(other_sections_task_error_real_matrix_results_data,'m_task_error_real_matrix_results')

#conect to df:
section_0_df = pd.concat([section_0_df, section_0_task_error_real_matrix_results_df], axis=1)
section_1_df = pd.concat([section_1_df, section_1_task_error_real_matrix_results_df], axis=1)
other_sections_df = pd.concat([other_sections_df, other_sections_task_error_real_matrix_results_df], axis=1)


##Task error - subject matrix:
#lode data
tasks_error_subject_matrix = pickle.load(open('data/tasks_error_subject_matrix', 'r'))

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
            task_error_subject_matrix_results[subject_id][step_id]=None

task_error_subject_matrix_results_df=pd.DataFrame.from_dict(task_error_subject_matrix_results, orient='index')
task_error_subject_matrix_results_df.drop(task_error_subject_matrix_results_df.columns[[8]], axis=1, inplace=True)


#section 0:
section_0_task_error_subject_matrix_results_df=pd.DataFrame(task_error_subject_matrix_results_df[1])
section_0_task_error_subject_matrix_results_df.columns.names=['subject_id']
section_0_task_error_subject_matrix_results_df.columns=['task_error_subject_matrix_results']

#section 1:
section_1_task_error_subject_matrix_results_df=pd.DataFrame(task_error_subject_matrix_results_df[2])
section_1_task_error_subject_matrix_results_df.columns.names=['subject_id']
section_1_task_error_subject_matrix_results_df.columns=['task_error_subject_matrix_results']

#all other sections:
other_sections_task_error_subject_matrix_results_data= pd.DataFrame(task_error_subject_matrix_results_df.iloc[:,2:])
other_sections_task_error_subject_matrix_results_df=linear_regression_from_df(other_sections_task_error_subject_matrix_results_data,'m_task_error_subject_matrix_results')

#conect to df:
section_0_df = pd.concat([section_0_df, section_0_task_error_subject_matrix_results_df], axis=1)
section_1_df = pd.concat([section_1_df, section_1_task_error_subject_matrix_results_df], axis=1)
other_sections_df = pd.concat([other_sections_df, other_sections_task_error_subject_matrix_results_df], axis=1)

##export to excel
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('data/big_analysis.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
section_0_df.to_excel(writer, sheet_name='section_0')
section_1_df.to_excel(writer, sheet_name='section_1')
other_sections_df.to_excel(writer, sheet_name='other_sections')

# Close the Pandas Excel writer and output the Excel file.
writer.save()