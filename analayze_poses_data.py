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

sections_list=['learn', 'task1', 'task2', 'task3']


##Delay analasis:
## Get error for each time stamp - error = skeleton * matrix - robot:
avg_error_per_delays=[]
for delay in range(0,4):


    poses = pickle.load(open('data_of_poses_'+str(delay), 'rb'))

    skeleton_metrix_robot_error={}
    for subject_id, step in poses.items():
        skeleton_metrix_robot_error[subject_id]={}
        which_matrix = int(subject_id) % 2
        for step_id, step in step.items():
            skeleton_metrix_robot_error[subject_id][step_id] = {}

            for section_id in poses[subject_id][step_id].keys():
                if section_id not in sections_list:
                    continue

                section=poses[subject_id][step_id][section_id]
                section_error=[]
                for i, d in enumerate(section['time']):
                    robot_calculation = np.dot(poses[subject_id][step_id]['matrix'], section['skeleton'][i])

                    error=np.linalg.norm((robot_calculation-section['robot'][i])[(0,1,4,5),])/4

                    section_error.append(error)

                skeleton_metrix_robot_error[subject_id][step_id][section_id] = {
                'time': section['time'],
                'error': section_error
                }

#todo- problem with matrix

#     #Data for plot:
#     avg_error_per_subject=[]
#     subject_id_for_plot=[]
#     for subject_id, sections in skeleton_metrix_robot_error.items():
#         avg_section=[]
#         for section_id, section in sections.items():
#             avg_section.append(np.nanmean(section['error']))
#
#         avg_error_per_subject.append(np.rad2deg(np.nanmean(avg_section)))
#         subject_id_for_plot.append(int(subject_id))
#     avg_error_per_delays.append(avg_error_per_subject)
#
# #Find time interval:
# intraval_time=[]
# data = pickle.load(open('raw_data', 'rb'))
# for subject_id, sections in data.items():
#     for section_id, section in sections.items():
#         if 'trans' not in section_id:
#             now=0
#             for step in section['data']:
#                 intraval= step['time'] -now
#                 intraval_time.append(intraval)
#                 now =step['time']
# intraval_time = round(np.median(intraval_time),2)
#
# #Plot
# data=[]
# for i in range(0,29):
#     lists=[[x, i*intraval_time] for x in avg_error_per_delays[i]]
#     [data.append(x) for x in lists]
# error=pd.DataFrame(data,columns=['error','delay'])
#
# for_rank= error.groupby(['delay'],as_index=False).mean()
# for_rank= for_rank['error']
#
# sns.set_style("whitegrid")
# pal = sns.color_palette("Blues_d", len(for_rank))
# rank = for_rank.argsort().argsort()
# ax = sns.barplot(x="delay", y="error", data=error, capsize=.2 ,palette=np.array(pal[::-1])[rank])
# ax.set(xlabel='Delay(sec)', ylabel='Avg Error (degrees)')
# sns.plt.title('Avg Error between robot angles and skeleton angles, in different delays')
# sns.plt.show()
