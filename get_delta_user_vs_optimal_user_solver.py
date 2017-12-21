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


def find_optimal_pose(_poses_list ,_true_matrix):
    poses_list=_poses_list
    n_pos=(len(poses_list))
    best_order = np.empty((0, 8))
    for t in range(n_pos):
        optimal_pose,optimal_index=find_next_pose(best_order,poses_list,_true_matrix)
        np.vstack((best_order, optimal_pose))
        del poses_list[optimal_index]
    return best_order


def find_next_pose(poses_list_previous,left_poses,_true_matrix):
    errors=[]
    for i in range((len(left_poses))):
        error=calc_matrix_error(poses_list_previous,left_poses[i],_true_matrix)
        errors.append(error)

    argmin=np.argmin(errors)
    return left_poses[argmin],argmin



# creating matrix error:

delta_user_vs_optimal_user={}
for subject_id, step in poses.items():

    delta_user_vs_optimal_user[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix']

        delta_user_vs_optimal_user[subject_id][step_id] = []

        for section_id in step.keys():

            if section_id=='learn':
                section=poses[subject_id][step_id][section_id]

                # skeleton_vectors=np.empty((0,8))
                #
                # robot_vectors =np.empty((0,8))


                real_poses=section['skeleton']



                # for i, d in enumerate(section['time']):
                #
                #
                #     skeleton_vectors=np.vstack((skeleton_vectors, section['skeleton'][i]))
                #
                #     _, locel_error = find_local_optimal_pose(skeleton_vectors, matrix)
                #
                #     print subject_id,step_id,i
                #
                #     local_optimal_pose[subject_id][step_id].append(locel_error)


pickle.dump(obj=delta_user_vs_optimal_user, file=open('data/delta_user_vs_optimal_user', 'wb'))



# for subject_id, matrix_step in local_optimal_pose.items():
#     for step_id, matrix_pose in matrix_step.items():
#         plt.plot(matrix_pose)
#         plt.title(str(subject_id) + ',' + str(step_id))
#         plt.show()