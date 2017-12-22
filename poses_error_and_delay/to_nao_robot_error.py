###Imports:
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from external_files.angle_matrix import AngleMatrix


sections_list=['learn', 'task1', 'task2', 'task3']

#Find time interval:
intraval_time=[]
data = pickle.load(open('/home/matan/PycharmProjects/physical_curiosity_big_analysis/data/raw_data_all_merged', 'rb'))

for subject_id, step in data.items():

    for step_id, step in step.items():

        for section_id in step.keys():
            if section_id not in sections_list:
                continue

            section=data[subject_id][step_id][section_id]

            now=0
            for intr in section['data']:
                intraval= intr['time'] -now

                intraval_time.append(intraval)

                now =intr['time']
intraval_time = round(np.median(intraval_time),2)

##Delay analasis:
## Get error for each time stamp - error = skeleton * matrix - robot:
avg_error_per_delays=[]
for delay in range(0,50):

    poses = pickle.load(open('/home/matan/PycharmProjects/physical_curiosity_big_analysis/data/data_of_poses_'+str(delay), 'rb'))

    skeleton_metrix_robot_error={}
    for subject_id, step in poses.items():
        skeleton_metrix_robot_error[subject_id]={}

        for step_id, step in step.items():
            skeleton_metrix_robot_error[subject_id][step_id] = {}

            for section_id in poses[subject_id][step_id].keys():
                if section_id not in sections_list:
                    continue

                section=poses[subject_id][step_id][section_id]
                section_error=[]
                for i, d in enumerate(section['time']):

                    error=np.linalg.norm((section['robot'][i]-section['to_nao'][i])[(0,1,4,5),])/4

                    section_error.append(error)

                skeleton_metrix_robot_error[subject_id][step_id][section_id] = {
                'time': section['time'],
                'error': section_error
                }


    #Data for plot:
    avg_error_per_subject=[]

    subject_id_for_plot=[]

    for subject_id, step in skeleton_metrix_robot_error.items():

        if subject_id == 16.0:
            continue

        avg_step=[]

        for step_id, step in step.items():

            for section_id in step.keys():

                if section_id not in sections_list:
                    continue

                section=skeleton_metrix_robot_error[subject_id][step_id][section_id]

                avg_step.append(np.nanmean(section['error']))

        avg_error_per_subject.append(np.rad2deg(np.nanmean(avg_step)))

        subject_id_for_plot.append(int(subject_id))

    avg_error_per_delays.append(avg_error_per_subject)



#Plot
data=[]
for i in range(0,50):
    lists=[[x, i*intraval_time] for x in avg_error_per_delays[i]]
    [data.append(x) for x in lists]
error=pd.DataFrame(data,columns=['error','delay'])

for_rank= error.groupby(['delay'],as_index=False).mean()
for_rank= for_rank['error']

sns.set_style("whitegrid")
pal = sns.color_palette("Blues_d", len(for_rank))
rank = for_rank.argsort().argsort()
ax = sns.barplot(x="delay", y="error", data=error, capsize=.2 ,palette=np.array(pal[::-1])[rank])
ax.set(xlabel='Delay(sec)', ylabel='Avg Error (degrees)')
ax.set_title('Avg Error between command to nao and robot angles, in different delays')
plt.show()
