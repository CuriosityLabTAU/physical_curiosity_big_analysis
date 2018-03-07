#imports:
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd

# load all the relevant data
# raw_data = pickle.load(open('data/raw_data_all_merged', 'r'))
# matrix_error_data = pickle.load(open('data/matrix_error_data', 'r'))
# optimal_user_error_sequence= pickle.load(open('data/optimal_user_error_sequence', 'r'))
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'r'))


# === data processing ===

# figure 1: how did we get poses
#--- this is done in the file: graph_for_poses.py ---

# figure 2: definitions of measures
# axis: x-axis p, y-axis matrix error
# draw for one subject for section two
# draw matrix error, and optimal error
# in PPT: mark b_global (min), b_local(area), b_sequence (area between)
def figure_2():
    step_n=2
    subject_id_n=8

    # collect matrix error data:
    matrix_error=[]
    for subject_id, step in matrix_error_data.items():
        if subject_id==subject_id_n:
            for step_id, errors in step.items():
                if step_id == step_n:
                    if 'error' in errors.keys():
                        if len(errors['error']) > 0:
                            matrix_error = errors['error']
                            break

    # collect matrix optimal error:
    optimal_error=optimal_user_error_sequence[subject_id_n][step_n]
    x=[i for i in range(len(matrix_error))]

    #plot
    # plt.figure()
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.plot(x, matrix_error ,label='Matrix error')
    plt.plot(x, optimal_error,label='Optimal error')

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error and optimal error - for one subject for section two')


    plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=5)

    plt.show()


# figure 3: what is matrix error
# axis: x-axis p, y-axis matrix error
# draw for all subjects (not average) for section 2
def figure_3():
    step_n=2

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    for subject_id, step in matrix_error_data.items():
            for step_id, errors in step.items():
                if step_id == step_n:
                    if 'error' in errors.keys():
                        if len(errors['error']) > 0:
                            y=errors['error']
                            x = [i for i in range(len(y))]

                            plt.plot(x, y)

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error and optimal error - for all subjects for section two')
    plt.ylim([0, 2])

    plt.show()


# figure 4: what is matrix error
# axis: x-axis p, y-axis matrix error
# draw average over participants, for all sections
def figure_4():

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # data collection
    steps=[0,2,4,5,7,8,10,11,12]
    step_dict={0:[], 2:[], 4:[], 5:[], 7:[], 8:[], 10:[], 11:[], 12:[]}
    for subject_id, step in matrix_error_data.items():
            for step_id, errors in step.items():
                    if 'error' in errors.keys():
                        if len(errors['error']) > 0:
                            step_dict[step_id].append(errors['error'])
    n = 1
    for step in steps:

        data=step_dict[step]
        length = len(sorted(data, key=len, reverse=True)[0])
        data_array = np.array([xi + [np.nan] * (length - len(xi)) for xi in data])

        y = np.nanmean(data_array, axis=0)
        x=[i for i in range(y.shape[0])]

        plt.plot(x, y,label='sections %d'%n)

        n+=1

    plt.ylabel('Error')
    plt.xlabel('Number of pose')
    plt.title('Matrix error - average over participants, for all sections')
    plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=5)
    plt.ylim([0, 1.5])
    plt.xlim([0, 15])
    plt.show()


# figure 5: how did we compute slopes
# axis: x-axis sessions 2-8, y-axis: for each measure X6
# draw for one subject (highest R^2), points and slope
def linear_regression_plot(data,_xlabel,_ylabel,_title):

    # start from 0:
    y = data - data[0]
    len_x = len(y)
    y = np.array(y)
    x = [i for i in range(len_x)]

    # crate x for a non intercepted linear regressio
    x = x[:, np.newaxis]

    # run linear regression
    m, _, _, _ = np.linalg.lstsq(x, y)

    #plot style:
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.plot(x, y, '.')
    plt.plot(x, m * x , '-')

    plt.ylabel(_ylabel)
    plt.xlabel(_xlabel)
    plt.title(_title)
    plt.show()


def figure_5():
    ## --Number of poses--
    subject_number_of_poses

    # crate df:
    subject_number_of_poses_df = pd.DataFrame.from_dict(subject_number_of_poses, orient='index')
    subject_number_of_poses_df.drop(subject_number_of_poses_df.columns[[1]], axis=1,
                                    inplace=True)  # delete the second epoch(no learning)
    other_sections_data = pd.DataFrame(subject_number_of_poses_df.iloc[:, 2:8])
    for i in range(other_sections_data.shape[0]):
        linear_regression_plot(other_sections_data.values[i],'section','number of poses','linear regression of number of poses in each section')

figure_5()

# === data analysis ====

# figure 6: measures' histograms
# 6 X 4 subplots
# each row is a section (1, 2, overline, 9)
# each column is a measure
# each subplot is a histogram of that measure


# figure 7: measures' correlation
# matrix/table: 24 X 24 (all measure X all measures
# Value is R, but only if p<0.05



# === internal insights ===
# figure 8: do participants who explore more learn more with time
# formula: (\delta^i | \tilde\delta) ~ matrix^i + i + (n_pose^i | b_...)


# === external validation ===
# formula '(CEI | BFI | PET) ~ factor(measures)'

# ad_hoc (figure 7) ==> 3 measures





