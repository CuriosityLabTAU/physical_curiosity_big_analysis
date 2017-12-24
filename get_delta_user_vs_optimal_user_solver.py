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

# poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer

# --- MATRIX: matrix * skeleton_vecotrs = robot_vectors

def calc_matrix_error(new_skeleton_vector, _skeleton_vectors, _matrix):
    # given skeleton vectors, calculate the true robot vectors from the true real matrix
    # optimally calculate the matrix from the skeleton vectors and true robot vectors
    # return the error

    for i in new_skeleton_vector:
        if i < -np.pi or i > np.pi:
            return np.inf

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


def find_optimal_error_sequence(_poses_list ,_true_matrix):
    poses_list=_poses_list
    n_pos=(len(poses_list))
    best_order = np.empty((0, 8))
    best_error_sequence=[]

    for t in range(n_pos):

        optimal_pose,optimal_index,last_error=find_next_pose(best_order,poses_list,_true_matrix)
        best_order=np.vstack((best_order, optimal_pose))
        del poses_list[optimal_index]
        best_error_sequence.append(last_error)

    return best_error_sequence


def find_next_pose(poses_list_previous,left_poses,_true_matrix):
    errors=[]
    for i in range((len(left_poses))):
        error=calc_matrix_error(left_poses[i],poses_list_previous,_true_matrix)
        errors.append(error)

    argmin=np.argmin(errors)

    return left_poses[argmin],argmin , errors[argmin]



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


                real_poses=section['skeleton']

                optimal_user_error[subject_id][step_id]=find_optimal_error_sequence(real_poses,matrix)




# pickle.dump(obj=delta_user_vs_optimal_user, file=open('data/delta_user_vs_optimal_user', 'wb'))

