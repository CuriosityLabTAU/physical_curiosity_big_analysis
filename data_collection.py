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
from sklearn import datasets, linear_model

# linear regression fun:
# def linear_regression_fron_df(data):
#     m=[]
#     c=[]
#     for row in subject_number_of_poses_df.iterrows():
#         print row[1].toarry()
#         break
#     m, c = np.linalg.lstsq(A, y)[0]
#
#
#     return df




##we will crate a DF with all the data:
#section 0 df
#section 1 df
#all other sections


## Number of poses
#lode data
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'rb'))

#crate df:
subject_number_of_poses_df=pd.DataFrame.from_dict(subject_number_of_poses, orient='index')

subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1, inplace=True)  #delete the second epoch(no learning)

#section 0 df
section_0_df=pd.DataFrame(subject_number_of_poses_df[0])
section_0_df.columns.names=['subject_id']
section_0_df.columns=['subject_number_of_poses']

#section 1 df
section_1_df=pd.DataFrame(subject_number_of_poses_df[2])
section_1_df.columns.names=['subject_id']
section_1_df.columns=['subject_number_of_poses']

#all other sections df
other_sections_data= pd.DataFrame(subject_number_of_poses_df.iloc[:,2:])

#linar model


# other_sections_df=