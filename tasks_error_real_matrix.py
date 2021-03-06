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

sections_list=['learn', 'task1', 'task2', 'task3']

#load data:
poses = pickle.load(open('data/data_of_poses_21', 'rb'))
# poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer


##Get task error (error = pose - task_pose):
task_error = {}
pass_threshold=20
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

                    error=0
                    task_pose_original=0
                    if section['task'] == 'two_hands_forward':
                        task_pose_original=np.dot(pinv(matrix), np.array([0.00, 0.00, 0.00, -0.034, 0.00, 0.00, 0.00, 0.034]))
                        error=pose-task_pose_original

                    elif section['task'] == 'two_hands_down':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.25, 0.00, 0.00, -0.034, 1.25, 0.00, 0.00, 0.034]))
                        error = pose - task_pose_original

                    elif section['task'] == 'two_hands_to_the_side':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.45, 1.30, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'two_hands_up':
                        task_pose_original = np.dot(pinv(matrix), np.array([-1.25, 0.0, 0.00 , -0.034, -1.25, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_down':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.25, 0.0, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_forward':
                        task_pose_original = np.dot(pinv(matrix), np.array([0.0, 0.0, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_up_left_hand_to_the_side':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.45, 1.3, 0.00, -0.034, -1.25, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_forward_left_hand_down':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.25, 0.00, 0.00, -0.034, 0.0, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_forward_left_hand_side':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.45, 1.3, 0.00, -0.034,  0.0, 0.0, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_to_the_side_left_hand_down':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.25, 0.00, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_to_the_side_left_hand_forward':
                        task_pose_original = np.dot(pinv(matrix), np.array([0.00, 0.00, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_down_left_hand_to_the_side':
                        task_pose_original = np.dot(pinv(matrix), np.array([1.45, 1.30, 0.00, -0.034, 1.25, 0.00, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'right_hand_down_left_hand_forward':
                        task_pose_original = np.dot(pinv(matrix), np.array([0.00, 0.00, 0.00, -0.034, 1.25, 0.00, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_down':
                        task_pose_original = np.dot(pinv(matrix), np.array([-1.25, 0.0, 0.00, -0.034, 1.25, 0.00, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_forward':
                        task_pose_original = np.dot(pinv(matrix), np.array([-1.25, 0.0, 0.00, -0.034, 0.00, 0.00, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    elif section['task'] == 'left_hand_up_right_hand_to_the_side':
                        task_pose_original = np.dot(pinv(matrix), np.array([-1.25, 0.0, 0.00, -0.034, 1.45, -1.30, 0.00, 0.034]))
                        error = (pose - task_pose_original)

                    if section['task']==0:
                        continue

                    agg_error=np.rad2deg((np.linalg.norm(error[(0,1,4,5),]))/4)

                    task_error[subject_id][step_id][section_id]['error'].append(agg_error)
                    task_error[subject_id][step_id][section_id]['min_error'] = min(task_error[subject_id][step_id][section_id]['error'])

pickle.dump(obj=task_error, file=open('data/tasks_error_real_matrix', 'wb'))

# #build pass task DF:
# task_results={}
# for subject_id, step in task_error.items():
#
#     task_results[subject_id] = {}
#
#     for step_id, step in step.items():
#
#         task_results[subject_id][step_id] = 0
#
#         step_results=[]
#
#         for section_id in step.keys():
#             if 'min_error' in task_error[subject_id][step_id][section_id].keys():
#
#                 if task_error[subject_id][step_id][section_id]['min_error'] <= pass_threshold:
#
#                     step_results.append(1)
#                 else:
#                     step_results.append(0)
#
#         if len(section_id)>0:
#             task_results[subject_id][step_id]=np.mean(step_results)
#
#
# task_results_df=pd.DataFrame.from_dict(task_results, orient='index')
#
# print list(task_results_df)
#
# df=task_results_df
#
#
#
# df.drop(df.columns[[0, 9]], axis=1, inplace=True)  # df.columns is zero-based pd.Index
# print df
#
# df=df.transpose()
#
# mean= df.mean(axis=1)
# print mean
#
# plt.figure()
# mean.plot()
# plt.show()
