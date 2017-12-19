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
from sklearn import linear_model


sections_list=['learn', 'task1', 'task2', 'task3']

poses = pickle.load(open('data/data_of_poses_21', 'rb')) #for lab computer
# poses = pickle.load(open('data/data_of_poses_21', 'r')) #for home computer


#Number of poses for subjects:

subject_number_of_poses={}

all_poses=np.empty((0,8))

for subject_id, step in poses.items():

    subject_number_of_poses[subject_id] = {}

    for step_id, step in step.items():

        subject_number_of_poses[subject_id][step_id]=0

        for section_id in step.keys():
            if section_id not in sections_list:
                continue

            section=poses[subject_id][step_id][section_id]

            for i, d in enumerate(section['time']):
                if section_id == 'learn':
                    subject_number_of_poses[subject_id][step_id] += 1

pickle.dump(obj=subject_number_of_poses, file=open('data/subject_number_of_poses', 'wb'))





# subject_number_of_poses_df=pd.DataFrame.from_dict(subject_number_of_poses, orient='index')
#
# subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1, inplace=True)  #delete the second epoch(no learning)
#
#
# median= subject_number_of_poses_df.median(numeric_only=True, axis=1)
# average = subject_number_of_poses_df.mean(numeric_only=True, axis=1)
#
# subject_number_of_poses_df['median'] =median
# subject_number_of_poses_df['average']=average
#
# print subject_number_of_poses_df


# #Linear Regression
# def linear_regression(x,y):
#     linear_model.LinearRegression().fit(x,y)
#
# x=[]
#
#
#
# # def linear_regression(x,y):
#
#
#
# #Todo: linear regretion
