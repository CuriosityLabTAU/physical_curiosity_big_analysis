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


# poses = pickle.load(open('data/data_of_poses_21', 'rb'))

poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer


def get_all_poses():
    all_poses = []
    for subject_id, step in poses.items():
        for step_id, step in step.items():
            for section_id in step.keys():
                if section_id == 'learn':
                    section = poses[subject_id][step_id][section_id]
                    for i, d in enumerate(section['time']):
                        all_poses.append(section['skeleton'][i])

    return all_poses

all_poses = np.array(get_all_poses())
all_poses_stats = {
    'min': np.min(all_poses, axis=0)[(0,1,4,5),],
    'max': np.max(all_poses, axis=0)[(0,1,4,5),],
    'mean': np.mean(all_poses, axis=0)[(0,1,4,5),]
}
print(all_poses_stats)



# --- MATRIX: matrix * skeleton_vecotrs = robot_vectors

def calc_matrix_error(new_skeleton_vector, _skeleton_vectors, _matrix):
    # given skeleton vectors, calculate the true robot vectors from the true real matrix
    # optimally calculate the matrix from the skeleton vectors and true robot vectors
    # return the error
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

# MATAN:do optimization
def find_optimal_pose(_true_matrix):
    poses_list = np.empty((0,8))
    errors = []
    for i in range(10):
        new_pose, error = find_local_optimal_pose(poses_list, _true_matrix)
        errors.append(error)
        poses_list = np.vstack((poses_list, new_pose))

    return poses_list, errors


def find_local_optimal_pose(poses_list_previous, _true_matrix):
    x0 = np.random.randn(4, 1) * np.pi / 6
    optimal_pose = scipy.optimize.minimize(fun=calc_matrix_error, x0=x0,
                                           args=(poses_list_previous, _true_matrix),
                                           bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                                                   (-np.pi, np.pi)],
                                           tol=1e-6)
    new_pose = [optimal_pose.x[0], optimal_pose.x[1], 0, 0,
                optimal_pose.x[2], optimal_pose.x[3], 0, 0]
    error = optimal_pose.fun
    return new_pose, error


# creating matrix error:

optimal_pose_error={}
local_optimal_pose={}
for subject_id, step in poses.items():

    optimal_pose_error[subject_id] = {}
    local_optimal_pose[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix']
        poses_list_optimal, errors_optimal = find_optimal_pose(matrix)

        optimal_pose_error[subject_id][step_id] = errors_optimal
        local_optimal_pose[subject_id][step_id] = errors_optimal

        for section_id in step.keys():

            if section_id=='learn':
                section=poses[subject_id][step_id][section_id]

                skeleton_vectors=np.empty((0,8))

                robot_vectors =np.empty((0,8))

                for i, d in enumerate(section['time']):
                    skeleton_vectors=np.vstack((skeleton_vectors, section['skeleton'][i]))
                    robot_vectors=np.vstack((robot_vectors, section['robot'][i]))

                    _, locel_error = find_local_optimal_pose(skeleton_vectors, matrix)

                    if i > 3:

                        pinv_skeleton = np.linalg.pinv(skeleton_vectors)
                        Amat = np.dot(pinv_skeleton, robot_vectors)

                        difference=matrix - Amat
                        difference=difference[(0,1,4,5),]
                        difference=difference[:,(0,1,4,5)]

                        error= np.linalg.norm(difference)/16
