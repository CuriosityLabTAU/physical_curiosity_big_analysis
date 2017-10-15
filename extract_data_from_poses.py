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

sections_list=['learn', 'task1', 'task2', 'task3']
poses = pickle.load(open('data_of_poses_21', 'rb'))


#Number of poses for subjects:

subject_number_of_poses={}

all_poses=np.empty((0,8))

for subject_id, step in poses.items():

    subject_number_of_poses[subject_id] = {}

    for step_id, step in step.items():

        subject_number_of_poses[subject_id][step_id]=0

        for section_id in step.keys():
            if section_id not in sections_list:
                continue

            section=poses[subject_id][step_id][section_id]

            for i, d in enumerate(section['time']):
                if section_id == 'learn':
                    subject_number_of_poses[subject_id][step_id] += 1

subject_number_of_poses_df=pd.DataFrame.from_dict(subject_number_of_poses, orient='index')

median= subject_number_of_poses_df.median(numeric_only=True, axis=1)
average = subject_number_of_poses_df.mean(numeric_only=True, axis=1)

subject_number_of_poses_df['median'] =median
subject_number_of_poses_df['average']=average

print subject_number_of_poses_df



# createing matrix error:

matrix_error = {}
for subject_id, step in poses.items():

    matrix_error[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix']

        matrix_error[subject_id][step_id] = []

        for section_id in step.keys():
            if section_id=='learn':
                section=poses[subject_id][step_id][section_id]

                skeleton_vectors=np.empty((0,8))

                robot_vectors =np.empty((0,8))
                for i, d in enumerate(section['time']):


                    skeleton_vectors=np.vstack((skeleton_vectors, section['skeleton'][i]))
                    robot_vectors=np.vstack((robot_vectors, section['robot'][i]))

                    if i > 3:
                        pinv_skeleton = np.linalg.pinv(skeleton_vectors)
                        Amat = np.dot(pinv_skeleton, robot_vectors)

                        difference=matrix - Amat
                        difference=difference[(0,1,4,5),]
                        difference=difference[:,(0,1,4,5)]

                        error= np.linalg.norm(difference)/4

                        # matrix_error[subject_id][step_id][i+1] = np.rad2deg(error)

                        matrix_error[subject_id][step_id].append(error)


last_matrix_error={}
parameter_a={}
parameter_b={}
parameter_c={}


# sns.set(color_codes=True)
# hist =sns.distplot(matrix_error_df, bins=9, kde=False, rug=True)
# hist.set(xlabel='Min Error (degrees)', ylabel='Number of subjects')
# plt.title('Histogram of matrix error across subjects')
# plt.show()

#statistical results - histogram of matrix_error across subjects:
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

for subject_id,step in matrix_error.items():
    last_matrix_error[subject_id] = {}
    parameter_a[subject_id] = {}
    parameter_b[subject_id] = {}
    parameter_c[subject_id] = {}

    for step_id, errors  in step.items():
        if len(errors) > 2 ###### :
            last_matrix_error[subject_id][step_id] = errors[-1]

            popt, pcov = curve_fit(func, [i for i in range(len(errors))], errors, maxfev = 30000)

            parameter_a[subject_id][step_id] = popt[0]
            parameter_b[subject_id][step_id] = popt[1]
            parameter_c[subject_id][step_id] = popt[2]

last_matrix_error_df=pd.DataFrame.from_dict(last_matrix_error, orient='index')
parameter_a_df=pd.DataFrame.from_dict(parameter_a, orient='index')
parameter_b_df=pd.DataFrame.from_dict(parameter_b, orient='index')
parameter_c_df=pd.DataFrame.from_dict(parameter_c, orient='index')

print last_matrix_error_df
print parameter_a_df
print parameter_b_df
print parameter_c_df



##Get task error (error = pose - task_pose):
task_error = {}
for subject_id, step in poses.items():

    task_error[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix']

        task_error[subject_id][step_id] = {}

        for section_id in step.keys():
            if section_id in ['task1', 'task2', 'task3']:

                section=poses[subject_id][step_id][section_id]
                task_error[subject_id][step_id][section_id]={'task':section['task'],'error':[]}

                for i, d in enumerate(section['time']):
                    pose = section['skeleton'][i]


###
                    if section['task'] not in ['two_hands_forward','two_hands_down','two_hands_to_the_side']:
                        print section['task']
                        continue
###

                    error=0
                    task_pose_original=0
                    if section['task'] == 'two_hands_forward':
                        task_pose_original=np.dot(np.array([0.00, 0.00, 0.00, -0.034, 0.00, 0.00, 0.00, 0.034]), pinv(matrix))
                        error=pose-task_pose_original

                    elif section['task'] == 'two_hands_down':
                        task_pose_original = np.dot(np.array([1.25, 0.00, 0.00, -0.034, 1.25, 0.00, 0.00, 0.034]), pinv(matrix))
                        error = pose - task_pose_original

                    elif section['task'] == 'two_hands_to_the_side':
                        task_pose_original = np.dot(np.array([1.45, 1.30, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]), pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'two_hands_up':
                        task_pose_original = np.dot(np.array([-1.25, 0.0, 0.00 , -0.034, -1.25, 0.0, 0.00, 0.034]), pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_down':
                        task_pose_original = np.dot(np.array([1.25, 0.0, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]), pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_forward':
                        task_pose_original = np.dot(np.array([0.0, 0.0, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_to_the_side':
                        task_pose_original = np.dot(np.array([1.45, 1.3, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_forward_left_hand_down':
                        task_pose_original = np.dot(np.array([1.25, 0.00, 0.00, -0.034, 0.0, 0.0, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_forward_left_hand_side':
                        task_pose_original = np.dot(np.array([1.45, 1.3, 0.00, -0.034,  0.0, 0.0, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_to_the_side_left_hand_down':
                        task_pose_original = np.dot(np.array([1.25, 0.00, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)



                    elif section['task'] == 'right_hand_to_the_side_left_hand_forward':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_down_left_hand_to_the_side':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_down_left_hand_forward':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_down':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_forward':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_to_the_side':
                        task_pose_original = np.dot(np.array([1.45, 1.00, 0.00, -0.034, 1.45, -1.00, 0.00, 0.034]),pinv(matrix))
                        error = (pose - task_pose_original)



                    agg_error=np.rad2deg(np.linalg.norm(error[(0,1,4,5),])/8)
                    task_error[subject_id][step_id][section_id]['error'].append(agg_error)
