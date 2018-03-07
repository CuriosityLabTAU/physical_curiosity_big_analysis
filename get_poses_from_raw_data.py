
###Imports:
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt
import seaborn as sns
import scipy.optimize
from numpy.linalg import inv
from sklearn.cluster import KMeans
import statsmodels.formula.api as sm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

###Parameters:
median_filter_window = 31
movement_threshold = 0.002

####subject 51 has no_angles_topic

### Get poses from data:
def get_poses(angles):
    t = angles[:, 0]
    # plt.plot(t, angles[:,:])

    f_angles = medfilt(angles[:, :], [median_filter_window, 1])
    # plt.plot(t, f_angles)

    d_angles = np.gradient(f_angles, axis=0)
    total_derivative = np.sum(d_angles ** 2, axis=1)
    # plt.plot(t, total_derivative)
    # plt.show()

    # binarize to movement/no-movement
    bin = np.argwhere(total_derivative < movement_threshold)
    bin = np.array([x[0] for x in bin])
    # print(bin)
    d_bin = bin[1:] - bin[:-1]
    # print(d_bin)

    # get start/stop of no movement
    start_stop = np.argwhere(d_bin > 1)
    no_movement_bins_start = np.array([x[0] for x in bin[start_stop[:-1]+1]])
    no_movement_bins_stop = np.array([x[0] for x in bin[start_stop[1:]]])
    # print(no_movement_bins_start, no_movement_bins_stop)
    # get angles of middle of no-movement section
    middle_bin = [(no_movement_bins_start[i] + no_movement_bins_stop[i])/2.0 for i in range(no_movement_bins_start.shape[0])]
    middle_bin = [int(x) for x in middle_bin]
    # print(middle_bin)

    pose = angles[middle_bin,:]
    return pose ,middle_bin

#Save skeleton robot poses, depending on delay:
data = pickle.load(open('data/raw_data_all_merged_2000', 'rb'))
for delay in range(21,22):
    print delay
    # data[id][step][section] = array(dict{skeleton, robot, time})
    poses = {}

    for subject_id, step in data.items():           # go over subject

        if subject_id==1.0 or subject_id==51.0:
            continue

        poses[subject_id] = {}
        print subject_id


        for step_id, step in step.items():    # go over steps
            if step_id in [3,6,9]:
                continue
            poses[subject_id][step_id]={}
            poses[subject_id][step_id]['transformation']=data[subject_id][step_id]['transformation']
            poses[subject_id][step_id]['matrix']=data[subject_id][step_id]['matrix']

            for section_id in ['learn', 'task1', 'task2', 'task3']:

                if section_id in data[subject_id][step_id].keys():

                    section=data[subject_id][step_id][section_id]

                    time_stamp = np.zeros([len(section['data']), 1])

                    skeleton_angles = np.zeros([len(section['data']), 8])

                    robot_angles=np.zeros([len(section['data']), 8])

                    to_nao=np.zeros([len(section['data']), 8])


                    affdex_data=[]

                    for i, d in enumerate(section['data']): # go over time-steps

                        time_stamp[i,0] = d['time']

                        skeleton_angles[i, :] = np.array([float(x) for x in d['skeleton'].split(',')])

                        to_nao[i, :] = np.array([float(x) for x in d['robot_cimmand'].split(';')[1].split(',')])

                        robot_angles[i, :] = np.array([float(x) for x in d['robot'].split(',')])

                        affdex_data.append(d['affdex'])

                    skeleton_poses, pose_bins = get_poses(skeleton_angles)


                   # skeleton_poses=skeleton_poses[:-3]
                   # pose_bins=pose_bins[:-3]

                    if len(pose_bins)>0:
                        while len(to_nao) < pose_bins[-1] + delay+1:
                            pose_bins.pop()
                            if len(pose_bins)==0:
                                break

                    to_nao = to_nao[[x+delay for x in pose_bins],:]

                    robot_poses = robot_angles[[x+delay for x in pose_bins],:]

                    time_stamp = time_stamp[pose_bins, :]

                    affdex = []

                    for i in pose_bins:
                        affdex.append(affdex_data[i])

                    poses[subject_id][step_id][section_id] = {
                        'time': time_stamp,
                        'skeleton': skeleton_poses,
                        'to_nao':to_nao,
                        'robot': robot_poses,
                        'affdex':affdex,
                        'task':data[subject_id][step_id][section_id]['task']
                    }



    pickle.dump(obj=poses, file=open('data_of_poses_'+str(delay), 'wb'))
