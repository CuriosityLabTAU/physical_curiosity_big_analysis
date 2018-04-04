# how to run to get all files:
# 1. number_of_poses.py
# 2. matrix_error.py
# 3. tasks_error_real_matrix.py
# 4. task_error_subject_matrix.py
# 5. get_gamma_user_vs_optimal_user_solver.py
# 8. all_data_file_builder.py
# 9. all_data_task_normalization.py
#10. all_data_matrix_normalization.py
#11. data_collection.py

import subprocess
# files_to_run=['number_of_poses.py','matrix_error.py']
files_to_run=[
              'number_of_poses.py',
              'matrix_error.py'
              'tasks_error_real_matrix.py',
              'task_error_subject_matrix.py',
              'get_gamma_user_vs_optimal_user_solver.py',
              'all_data_file_builder.py',
              'all_data_task_normalization.py',
              'all_data_matrix_normalization.py',
              'data_collection.py'
              ]


for file in files_to_run:
    cmd = ['python', file]
    print "~~~~~~~~",file,"~~~~~~~~"
    subprocess.Popen(cmd).wait()