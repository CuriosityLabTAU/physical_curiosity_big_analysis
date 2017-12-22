###Imports:
import pickle



data1 = pickle.load(open('data/raw_data_all', 'rb'))
data2 = pickle.load(open('data/raw_data_all_5', 'rb'))

subject_id_data1 = []
subject_id_data2 = []

for subject_id, step in data1.items():
    subject_id_data1.append(subject_id)

for subject_id, step in data2.items():
    subject_id_data2.append(subject_id)

intersection = [x for x in subject_id_data1 if x in subject_id_data2]

map(data2.__delitem__, filter(data2.__contains__,intersection)) #delete form data2 the intersection

data1.update(data2) #merge




pickle.dump(obj=data1, file=open('raw_data_all_merged', 'wb'))
