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
import scipy.optimize


poses = pickle.load(open('data/data_of_poses_21', 'rb'))
matrix_error = pickle.load(open('data/matrix_error_data', 'rb'))

# poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer

#todo : here - change culcolation
def calc_matrix_error(new_skeleton_vector, _skeleton_vectors, _matrix):
    skeleton_full = [new_skeleton_vector[0], new_skeleton_vector[1], 0, 0,
                     new_skeleton_vector[2], new_skeleton_vector[3], 0, 0]

    _skeleton_vectors_full = np.vstack((_skeleton_vectors, skeleton_full))
    pinv_skeleton = np.linalg.pinv(_skeleton_vectors_full)
    true_robot_vectors = np.dot(_skeleton_vectors_full, _matrix)
    Amat = np.dot(pinv_skeleton, true_robot_vectors)

    difference = _matrix - Amat
    difference = difference[(0, 1, 4, 5),]
    difference = difference[:, (0, 1, 4, 5)]

    error = np.linalg.norm(difference) / 16
    return error


def find_optimal_error_sequence(_real_poses_skeleton,_poses_list_robot,_true_matrix):
    real_poses_skeleton, poses_list_robot = _real_poses_skeleton,_poses_list_robot
    n_pos=(len(real_poses_skeleton))

    best_order_skeleton = np.empty((0, 8))
    best_order_robot = np.empty((0, 8))

    best_error_sequence=[]

    for t in range(n_pos):

        optimal_pose_skeleton,optimal_pose_robot,optimal_index,last_error=find_next_pose(best_order_skeleton,best_order_robot,real_poses_skeleton, poses_list_robot,_true_matrix)
        best_order_skeleton=np.vstack((best_order_skeleton, optimal_pose_skeleton))
        best_order_robot=np.vstack((best_order_robot, optimal_pose_robot))
        real_poses_skeleton=np.delete(real_poses_skeleton, optimal_index, 0)
        poses_list_robot=np.delete(poses_list_robot, optimal_index, 0)

        best_error_sequence.append(last_error)

    return best_error_sequence


def find_next_pose(poses_list_previous_skeleton,poses_list_previous_robot,left_poses_skeleton,left_poses_robot,_true_matrix):
    errors=[]
    for i in range((len(left_poses_skeleton))):
        error=calc_matrix_error(left_poses_skeleton[i],left_poses_robot[i],poses_list_previous_skeleton,poses_list_previous_robot, _true_matrix)
        errors.append(error)

    argmin=np.argmin(errors)

    return left_poses_skeleton[argmin],poses_list_previous_robot[argmin],argmin , errors[argmin]



# creating matrix error:

optimal_user_error={}
for subject_id, step in poses.items():

    optimal_user_error[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix']

        optimal_user_error[subject_id][step_id] = []

        for section_id in step.keys():

            if section_id=='learn':
                section=poses[subject_id][step_id][section_id]


                real_poses_skeleton = section['skeleton']
                real_poses_robot = section['robot']


                optimal_user_error[subject_id][step_id]=find_optimal_error_sequence(real_poses,matrix)

                print 'hhh'




# pickle.dump(obj=delta_user_vs_optimal_user, file=open('data/delta_user_vs_optimal_user', 'wb'))

