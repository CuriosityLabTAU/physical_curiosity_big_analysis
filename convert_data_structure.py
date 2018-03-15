import pickle
import json

optimal_user_error_sequence= pickle.load(open('data/optimal_user_error_sequence', 'r'))
matrix_error_data=pickle.load(open('data/matrix_error_data', 'r'))
subject_number_of_poses = pickle.load(open('data/subject_number_of_poses', 'r'))
tasks_error_real_matrix = pickle.load(open('data/tasks_error_real_matrix', 'r'))
tasks_error_subject_matrix = pickle.load(open('data/tasks_error_subject_matrix', 'r'))
gamma_optimal_user_error_df = pickle.load(open('data/gamma_user_vs_optimal_user', 'r'))


pickle.dump(optimal_user_error_sequence, file=open('data/goren/optimal_user_error_sequence_goren', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(matrix_error_data, file=open('data/goren/matrix_error_data_goren', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(subject_number_of_poses, file=open('data/goren/subject_number_of_poses_goren', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(tasks_error_real_matrix, file=open('data/goren/tasks_error_real_matrix_goren', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(tasks_error_subject_matrix, file=open('data/goren/tasks_error_subject_matrix_goren', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
gamma_optimal_user_error_df.to_csv('data/goren/gamma_optimal_user_error_df_goren.csv')