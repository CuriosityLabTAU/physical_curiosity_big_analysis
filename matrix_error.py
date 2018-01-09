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

poses = pickle.load(open('data/data_of_poses_21', 'rb'))

# poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer

# createing matrix error:
matrix_error = {}
for subject_id, step in poses.items():

    matrix_error[subject_id] = {}

    for step_id, step in step.items():

        matrix=poses[subject_id][step_id]['matrix'][(0,1,4,5),]
        matrix=matrix[:,(0,1,4,5)]


        for section_id in step.keys():

            if section_id=='learn':

                matrix_error[subject_id][step_id] = {}
                matrix_error[subject_id][step_id]['error'] = []
                matrix_error[subject_id][step_id]['matrix'] = []

                section=poses[subject_id][step_id][section_id]

                skeleton_vectors=np.empty((0,4))

                robot_vectors =np.empty((0,4))


                for i, d in enumerate(section['time']):
                    skeleton_vectors=np.vstack((skeleton_vectors, section['skeleton'][i][(0,1,4,5),]))
                    robot_vectors=np.vstack((robot_vectors, section['robot'][i][(0,1,4,5),]))

                    if i < 3:
                        continue

                    pinv_skeleton = np.linalg.pinv(skeleton_vectors)
                    # Amat = np.dot(pinv_skeleton, robot_vectors)
                    Amat = np.dot(robot_vectors.T, pinv_skeleton.T)


                    difference=matrix - Amat

                    error= np.linalg.norm(difference)/16

                    matrix_error[subject_id][step_id]['matrix'].append(Amat)
                    matrix_error[subject_id][step_id]['error'].append(error)


                #NO Data in the the matrix_error[subject_id][step_id]['error']
                if len(matrix_error[subject_id][step_id]['error'])==0:
                    matrix_error[subject_id][step_id]['min_error'] = None
                    matrix_error[subject_id][step_id]['best_matrix'] = None

                else:
                    argmin_for_best_error=np.argmin(matrix_error[subject_id][step_id]['error'])
                    matrix_error[subject_id][step_id]['min_error']=matrix_error[subject_id][step_id]['error'][argmin_for_best_error]
                    matrix_error[subject_id][step_id]['best_matrix']=matrix_error[subject_id][step_id]['matrix'][argmin_for_best_error]

pickle.dump(obj=matrix_error, file=open('data/matrix_error_data', 'wb'))

for subject_id, matrix_step in matrix_error.items():
    for step_id, matrix_pose in matrix_step.items():
        plt.plot(matrix_pose['error'])
        plt.title(str(subject_id) + ',' + str(step_id))
        plt.show()


# last_matrix_error={}
# parameter_a={}
# parameter_b={}
# parameter_c={}


# sns.set(color_codes=True)
# hist =sns.distplot(matrix_error_df, bins=9, kde=False, rug=True)
# hist.set(xlabel='Min Error (degrees)', ylabel='Number of subjects')
# plt.title('Histogram of matrix error across subjects')
# plt.show()

#statistical results - histogram of matrix_error across subjects:
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# for subject_id,step in matrix_error.items():
#     last_matrix_error[subject_id] = {}
#     parameter_a[subject_id] = {}
#     parameter_b[subject_id] = {}
#     parameter_c[subject_id] = {}
#
#     for step_id, errors  in step.items():
#         if len(errors) > 2: ######
#             last_matrix_error[subject_id][step_id] = errors[-1]
#
#             popt, pcov = curve_fit(func, [i for i in range(len(errors))], errors, maxfev = 30000)
#
#             parameter_a[subject_id][step_id] = popt[0]
#             parameter_b[subject_id][step_id] = popt[1]
#             parameter_c[subject_id][step_id] = popt[2]
#
# last_matrix_error_df=pd.DataFrame.from_dict(last_matrix_error, orient='index')
# parameter_a_df=pd.DataFrame.from_dict(parameter_a, orient='index')
# parameter_b_df=pd.DataFrame.from_dict(parameter_b, orient='index')
# parameter_c_df=pd.DataFrame.from_dict(parameter_c, orient='index')
#
# print last_matrix_error_df
# print parameter_a_df
# print parameter_b_df
# print parameter_c_df


