#imports
import pandas as pd



all_data=pd.read_excel("data/all_data.xlsx")

steps=[0,1,2,3,4,5,6,7,8]    #step_id
matrices=[0,1,2,3,4,5,6,7,8] #matrix
measures=['min_matrix_error','sum_matrix_error', 'task_error_real_matrix', 'task_error_subject_matrix', 'behavior_gamma']


for measure in measures:
    for mat in matrices:
        ind=all_data.index[all_data['matrix'] == mat].tolist()

        #normalization method
        all_data.loc[ind, measure] = all_data.loc[ind, measure].values - all_data.loc[ind, measure].values.mean()

##export to excel
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('data/all_data_normalized.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
all_data.to_excel(writer, sheet_name='all_data_normalized')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
