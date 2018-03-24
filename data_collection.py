###Imports:
import pickle
import numpy as np
import pandas as pd
import matplotlib


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


def data_collection(_in,_out):
    all_data = pd.read_excel(_in)

    ##we will crate a DF with all the data:
    #section 0 df
    #section 1 df
    #all other sections
    #section 9


    ## Number of poses
    subject_number_of_poses_df=all_data.pivot(index='subject_id', columns='step_id')['number_of_poses']
    subject_number_of_poses_df.rename_axis(None)

    #section 0 df
    section_0_df=pd.DataFrame(subject_number_of_poses_df[0])
    section_0_df.columns.names=['subject_id']
    section_0_df.columns=['subject_number_of_poses_0']

    #section 1 df
    section_1_df=pd.DataFrame(subject_number_of_poses_df[1])
    section_1_df.columns.names=['subject_id']
    section_1_df.columns=['subject_number_of_poses_1']

    #section 9 df
    section_9_df=pd.DataFrame(subject_number_of_poses_df[8])
    section_9_df.columns.names=['subject_id']
    section_9_df.columns=['subject_number_of_poses_9']

    #all other sections df
    other_sections_data= pd.DataFrame(subject_number_of_poses_df.iloc[:,2:8])
    other_sections_df=linear_regression_from_df(other_sections_data,'m_number_of_poses')


    ##Matrix error:
    #lode data
    matrix_error = pickle.load(open('data/matrix_error_data', 'r'))

    min_matrix_error=all_data.pivot(index='subject_id', columns='step_id')['min_matrix_error']
    min_matrix_error.rename_axis(None)

    sum_matrix_error=all_data.pivot(index='subject_id', columns='step_id')['sum_matrix_error']
    sum_matrix_error.rename_axis(None)

    #section 0:
    section_0_min_matrix_df=pd.DataFrame(min_matrix_error[0])
    section_0_min_matrix_df.columns.names=['subject_id']
    section_0_min_matrix_df.columns=['min_matrix_error_0']
    section_0_sum_matrix_df=pd.DataFrame(sum_matrix_error[0])
    section_0_sum_matrix_df.columns.names=['subject_id']
    section_0_sum_matrix_df.columns=['sum_matrix_error_0']

    #section 1:
    section_1_min_matrix_df=pd.DataFrame(min_matrix_error[1])
    section_1_min_matrix_df.columns.names=['subject_id']
    section_1_min_matrix_df.columns=['min_matrix_error_1']
    section_1_sum_matrix_df=pd.DataFrame(sum_matrix_error[1])
    section_1_sum_matrix_df.columns.names=['subject_id']
    section_1_sum_matrix_df.columns=['sum_matrix_error_1']

    #section 9:
    section_9_min_matrix_df=pd.DataFrame(min_matrix_error[8])
    section_9_min_matrix_df.columns.names=['subject_id']
    section_9_min_matrix_df.columns=['min_matrix_error_9']
    section_9_sum_matrix_df=pd.DataFrame(sum_matrix_error[8])
    section_9_sum_matrix_df.columns.names=['subject_id']
    section_9_sum_matrix_df.columns=['sum_matrix_error_9']

    #all other sections:
    other_sections_min_matrix_data= pd.DataFrame(min_matrix_error.iloc[:,2:8])
    other_sections_min_matrix_df=linear_regression_from_df(other_sections_min_matrix_data,'m_min_matrix_error')

    other_sections_sum_matrix_data= pd.DataFrame(sum_matrix_error.iloc[:,2:8])
    other_sections_sum_matrix_df=linear_regression_from_df(other_sections_sum_matrix_data,'m_sum_matrix_error')

    #conect to df:
    section_0_df = pd.concat([section_0_df, section_0_min_matrix_df,section_0_sum_matrix_df], axis=1)
    section_1_df = pd.concat([section_1_df, section_1_min_matrix_df,section_1_sum_matrix_df], axis=1)
    section_9_df = pd.concat([section_9_df, section_9_min_matrix_df,section_9_sum_matrix_df], axis=1)
    other_sections_df = pd.concat([other_sections_df, other_sections_min_matrix_df,other_sections_sum_matrix_df], axis=1)


    ##Task error - real matrix:
    #lode data
    task_error_real_matrix_results_df=all_data.pivot(index='subject_id', columns='step_id')['task_error_real_matrix']
    task_error_real_matrix_results_df.rename_axis(None)

    #section 0:
    section_0_task_error_real_matrix_results_df=pd.DataFrame(task_error_real_matrix_results_df[0])
    section_0_task_error_real_matrix_results_df.columns.names=['subject_id']
    section_0_task_error_real_matrix_results_df.columns=['task_error_real_matrix_results_0']

    #section 1:
    section_1_task_error_real_matrix_results_df=pd.DataFrame(task_error_real_matrix_results_df[1])
    section_1_task_error_real_matrix_results_df.columns.names=['subject_id']
    section_1_task_error_real_matrix_results_df.columns=['task_error_real_matrix_results_1']


    #all other sections:
    other_sections_task_error_real_matrix_results_data= pd.DataFrame(task_error_real_matrix_results_df.iloc[:,2:8])
    other_sections_task_error_real_matrix_results_df=linear_regression_from_df(other_sections_task_error_real_matrix_results_data,'m_task_error_real_matrix_results')

    #conect to df:
    section_0_df = pd.concat([section_0_df, section_0_task_error_real_matrix_results_df], axis=1)
    section_1_df = pd.concat([section_1_df, section_1_task_error_real_matrix_results_df], axis=1)
    other_sections_df = pd.concat([other_sections_df, other_sections_task_error_real_matrix_results_df], axis=1)


    ##Task error - subject matrix:
    #lode data
    task_error_subject_matrix_results_df=all_data.pivot(index='subject_id', columns='step_id')['task_error_subject_matrix']
    task_error_subject_matrix_results_df.rename_axis(None)


    #section 0:
    section_0_task_error_subject_matrix_results_df=pd.DataFrame(task_error_subject_matrix_results_df[0])
    section_0_task_error_subject_matrix_results_df.columns.names=['subject_id']
    section_0_task_error_subject_matrix_results_df.columns=['task_error_subject_matrix_results_0']

    #section 1:
    section_1_task_error_subject_matrix_results_df=pd.DataFrame(task_error_subject_matrix_results_df[1])
    section_1_task_error_subject_matrix_results_df.columns.names=['subject_id']
    section_1_task_error_subject_matrix_results_df.columns=['task_error_subject_matrix_results_1']

    #all other sections:
    other_sections_task_error_subject_matrix_results_data= pd.DataFrame(task_error_subject_matrix_results_df.iloc[:,2:8])
    other_sections_task_error_subject_matrix_results_df=linear_regression_from_df(other_sections_task_error_subject_matrix_results_data,'m_task_error_subject_matrix_results')

    #conect to df:
    section_0_df = pd.concat([section_0_df, section_0_task_error_subject_matrix_results_df], axis=1)
    section_1_df = pd.concat([section_1_df, section_1_task_error_subject_matrix_results_df], axis=1)
    other_sections_df = pd.concat([other_sections_df, other_sections_task_error_subject_matrix_results_df], axis=1)



    ## Behavior gamma
    #lode data
    gamma_optimal_user_error_df=all_data.pivot(index='subject_id', columns='step_id')['behavior_gamma']
    gamma_optimal_user_error_df.rename_axis(None)

    #section 0 df
    behavior_gamma_0_df=pd.DataFrame(gamma_optimal_user_error_df[0])
    behavior_gamma_0_df.columns.names=['subject_id']
    behavior_gamma_0_df.columns=['behavior_gamma_0']

    #section 1 df
    behavior_gamma_1_df=pd.DataFrame(gamma_optimal_user_error_df[1])
    behavior_gamma_1_df.columns.names=['subject_id']
    behavior_gamma_1_df.columns=['behavior_gamma_1']

    #section 9 df
    behavior_gamma_9_df=pd.DataFrame(gamma_optimal_user_error_df[8])
    behavior_gamma_9_df.columns.names=['subject_id']
    behavior_gamma_9_df.columns=['behavior_gamma_9']

    #all other sections df
    other_sections_gamma= pd.DataFrame(gamma_optimal_user_error_df.iloc[:,2:8])
    behavior_gamma_sections_df=linear_regression_from_df(other_sections_gamma,'m_behavior_gamma')

    #conect to df:
    section_0_df = pd.concat([section_0_df, behavior_gamma_0_df], axis=1)
    section_1_df = pd.concat([section_1_df, behavior_gamma_1_df], axis=1)
    section_9_df = pd.concat([section_9_df, behavior_gamma_9_df], axis=1)
    other_sections_df = pd.concat([other_sections_df, behavior_gamma_sections_df], axis=1)

    ##export to excel
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(_out, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    section_0_df.to_excel(writer, sheet_name='section_0')
    section_1_df.to_excel(writer, sheet_name='section_1')
    section_9_df.to_excel(writer, sheet_name='section_9')
    other_sections_df.to_excel(writer, sheet_name='other_sections')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


data_collection("data/all_data.xlsx",'data/big_analysis_n.xlsx')
data_collection("data/all_data_normalized.xlsx",'data/big_analysis_normalized.xlsx')