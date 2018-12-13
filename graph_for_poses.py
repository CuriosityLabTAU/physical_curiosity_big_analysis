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
import rosbag
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
bridge=CvBridge()
###Parameters:
median_filter_window = 31
movement_threshold = 0.002
import seaborn as sns
import matplotlib.gridspec as gridspec
sns.set_style("darkgrid", {"axes.facecolor": "#DDE5EE"})
from matplotlib.ticker import MaxNLocator

def get_poses(angles,time):
    # t = [i for i in range(angles.shape[0])]
    time=time-min(time)
    t= time
    # plt.plot(t, angles[:,0:])

    f_angles = medfilt(angles[:, :], [median_filter_window, 1])
    f, ax = plt.subplots(1, 1,figsize=(8,3))

    d_angles = np.gradient(f_angles, axis=0)
    total_derivative = np.sum(d_angles ** 2, axis=1)
    ax.plot(t, total_derivative,label='total derivative')
    horiz_line_data = np.array([movement_threshold for i in xrange(t.shape[0])])
    ax.plot(t, horiz_line_data, 'r--',label='movement threshold')
    plt.ylim([-0.001, 0.01])
    plt.xlim([30, 40])
    plt.ylabel('Total derivative '+r'($radians^2$)', fontsize='24') # for legend text

    plt.xlabel('Time (sec)',fontsize='24')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),facecolor='white',frameon=False)
    plt.setp(ax.get_legend().get_texts(), fontsize='24') # for legend text

    plt.show()


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

    # #plot
    # plt.plot(t, f_angles[:,0],label='Left shoulder joint front and back')
    # plt.plot(t, f_angles[:,1],label='Left shoulder joint right and left')
    # plt.plot(t, f_angles[:,4],label='Right shoulder joint front and back')
    # plt.plot(t, f_angles[:,5],label='Right shoulder joint right and left')
    #


    #
    # plt.ylabel('Angles (radians)')
    # plt.xlabel('Time (sec)')
    #
    # plt.xlim([30, 40])
    #
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
    # plt.show()

    return pose ,middle_bin


data = pickle.load(open('data/new_data/raw_data_all_merged', 'r'))

for subject_id, step in data.items():

    if subject_id == 23:


        for step_id, step in step.items():
            if step_id==4:

                section = data[subject_id][step_id]['learn']

                time_stamp = np.zeros([len(section['data']), 1])

                skeleton_angles = np.zeros([len(section['data']), 8])


                for i, d in enumerate(section['data']):

                    time_stamp[i, 0] = d['time']

                    skeleton_angles[i, :] = np.array([float(x) for x in d['skeleton'].split(',')])

pose ,middle_bin = get_poses(skeleton_angles,time_stamp)
time_of_bins=time_stamp[middle_bin].tolist()
time_of_bins = [ round(elem[0], 1) for elem in time_of_bins ]




#
# count=0
#
# sections = ['learn', 'task1', 'task2', 'task3']
#
# affdex_list=['emotions','expressions','measurements','face_points','face_x_alignment',
#              'face_y_alignment','face_tracking_deltax','face_distance','face_detected','tega_affect_msg','attention_detected']
#
# mypath = '/home/matan/Desktop/data/23/'
#
# files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.bag' in f]
# subjects_with_no_angles_topic=[]
# data = {}
# # for each bag file
# for f in files:
#     info = f.split('_')
#     subject_id = float(info[4])
#     print subject_id
#
#     if subject_id > 0.0:
#         print('processing ', subject_id)
#         count+=1
#         print "count= ",count
#         data[subject_id] = {}
#         data[subject_id][0] = {}
#
#
#         # open the bag file
#         bag = rosbag.Bag(mypath + f)
#
#         # the sections:
#         section_id = 0
#
#         #the task:
#         task=0
#
#         #the step:
#         step=0
#
#         #affdex:
#         affdex=0
#
#
#         #is_nao_angles_topic
#         is_nao_angles_topic=False
#
#         is_working= False
#
#         current_skeleton_angle = None
#         current_nao_movements = None
#         current_nao_command = None
#
#         first_time = True
#
#         # try:
#
#         for topic, msg, t in bag.read_messages():
#             # get the first time, for reference
#             if first_time:
#                 t0 = t
#                 first_time = False
#
#             if 'log' in topic:
#                 if'current_state' in msg.data:
#                     step=int(msg.data.split(': ')[1])
#                     if step in data[subject_id]:
#                         continue
#
#                     data[subject_id][step]={}
#                     if step == 1:
#                         section_id = 1
#                         print 'here'
#                     else:
#                         section_id = 0
#
#                 if 'matrix' in msg.data:
#                     if step in data[subject_id]:
#                         matrix=msg.data.split(':')[1]
#                         data[subject_id][step]['matrix'] = np.array(np.matrix([[float(x)for x in row if x!=''] for row in [line[2:-1].split(' ') for line in  matrix[2:-1].split('\n ')]]))
#
#
#                 if'task' in msg.data:
#                     task_full_name=msg.data.split(',')[2]
#                     task=task_full_name.split('/')[-1]
#
#
#             if 'flow' in topic:
#                 #   get the transformation
#
#                 if msg.data.isdigit():
#                     if step in data[subject_id]:
#                         data[subject_id][step]['transformation'] = int(msg.data)
#
#                   # parse the time to sections (learning, task1, task2, task3)
#                 if 'start' in msg.data:
#                     if step in data[subject_id]:
#                         is_working=True
#                         data[subject_id][step][sections[section_id]] = {
#                             'start': t,
#                             'stop': None,
#                             'data': [],
#                             'task':task
#                         }
#
#                 if 'stop' in msg.data:
#                     if step in data[subject_id]:
#                         if sections[section_id] in data[subject_id][step]:
#                             is_working=False
#                             data[subject_id][step][sections[section_id]]['stop'] = t
#                             task = 0
#                             section_id += 1
#
#                             if step == 12:
#                                 bag.close()
#                                 break
#
#                 if section_id >= len(sections):
#                     section_id=0
#
#
#             if step==4:
#                 if section_id==0:
#                     if topic== '/usb_cam/image_raw':
#                         if (t - t0).to_sec() < (40+time_stamp[0]) and (t - t0).to_sec() > (30+time_stamp[0]):
#                             print "here"
#                             if round((t - t0).to_sec(), 1) in time_of_bins:
#                                 cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
#                                 plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
#                                 plt.show()
#
#             if step==5:
#                 break

