###Imports:
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.optimize

nuber_of_samples=100

###crate dara using matrix:
#get matrix from data:
poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer
matrix=0
for subject_id, step in poses.items():
    for step_id, step in step.items():
        matrix=poses[subject_id][step_id]['matrix']
        break
    break

#crate dara:
samples=np.random.random((nuber_of_samples,8))*np.pi
data=np.dot(samples,matrix)


#data for plot:
x=[i+3 for i in range(98)]
errors=[]

#use matrix algorithm to reproduce matrix
skeleton_vectors = np.empty((0, 8))

robot_vectors = np.empty((0, 8))
for i in range(nuber_of_samples):

    skeleton_vectors = np.vstack((skeleton_vectors, samples[i]))
    robot_vectors = np.vstack((robot_vectors,  data[i]))

    if i>1:
        pinv_skeleton = np.linalg.pinv(skeleton_vectors)
        Amat = np.dot(pinv_skeleton, robot_vectors)

        difference = matrix - Amat
        difference = difference[(0, 1, 4, 5),]
        difference = difference[:, (0, 1, 4, 5)]

        error = np.linalg.norm(difference) / 16
        errors.append(error)


#plot error:
plt.figure()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plt.plot(x, errors, 'b.')
plt.xlabel('number of samples')
plt.ylabel('error')
plt.title('Matrix algorithm validation')
plt.show()

