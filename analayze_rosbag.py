# get a list of all the relevant bag files


#   for each section
#       get raw skeleton markers
#       get command to robot
#       dict[id] = dict[section] = array(dict{skeleton, robot, time})


import rosbag
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np

count=0
sections = ['learn', 'task1', 'task2', 'task3']

mypath = '/home/matan/Desktop/data/'

files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.bag' in f]

data = {}
# for each bag file
for f in files:
    info = f.split('_')
    subject_id = float(info[4])
    print subject_id
    if subject_id > 0.0:
        print('processing ', subject_id)
        count+=1
        print "count= ",count
        data[subject_id] = {}

        # open the bag file
        bag = rosbag.Bag(mypath + f)

        # the sections:
        section_id = 0

        #the task:
        task=0

        #the step:
        step=0

        #affdex:
        affdex=0

        current_skeleton_angle = None
        current_nao_movements = None

        first_time = True
        # try:

        for topic, msg, t in bag.read_messages():
            # get the first time, for reference
            if first_time:
                t0 = t
                first_time = False

            if 'log' in topic:
                if'current_state' in msg.data:
                    step=msg.data.split(':')[1]
                    data[subject_id][step]={}

                if 'matrix' in msg.data:
                    matrix=msg.data.split(':')[1]
                    data[subject_id][step]['matrix'] = np.array(matrix)

                if'task' in msg.data:
                    task_full_name=msg.data.split(',')[2]
                    task=task_full_name.split('/')[-1]


            if 'flow' in topic:
                #   get the transformation

                if msg.data.isdigit():
                    data[subject_id][step]['transformation'] = int(msg.data)

                  # parse the time to sections (learning, task1, task2, task3)
                if 'start' in msg.data:
                    print subject_id,step,sections[section_id]
                    print task,t
                    data[subject_id][step][sections[section_id]] = {
                        'start': t,
                        'stop': None,
                        'data': [],
                        'task':task
                    }

                if 'stop' in msg.data:
                    data[subject_id][step][sections[section_id]]['stop'] = t
                    task = 0
                    section_id += 1
                    print 'here'

                    if step == 12:
                        bag.close()
                        break

                if section_id >= len(sections):
                    section_id=0


            # if 'affdex' in topic:
            #     affdex= msg


              # for each section
            # if 'nao_movements' in topic:
#                #          get command to robot
                # current_nao_movements = msg.data


            # if 'skeleton_angle' in topic:
            #     #       get raw skeleton markers
            #     current_skeleton_angle = msg.data

            #
            # if 'nao_angles_topic' in topic:
            #     #       get command to robot
            #     list_of_movements_str=msg.data
            #     list_of_movements=list_of_movements_str[1:-1].split(', ')
            #
            #
            #     current_nao_movements = (list_of_movements[2:4]+[0.0,0.0]+list_of_movements[-6:-4]+[0.0,0.0])
            #
            #     current_nao_movements= str([float(x) for x in current_nao_movements])
            #     current_nao_movements= current_nao_movements[1:-1]


                #       dict[id] = dict[section] = array(dict{skeleton, robot, time})
                # if sections[section_id] in data[subject_id][step]:
                #     if current_skeleton_angle is not None and current_nao_movements is not None:
                #         new_data = {
                #             'time': (t - t0).to_sec(),
                #             'skeleton': current_skeleton_angle,
                #             'robot': current_nao_movements,
                #             'affdex':affdex
                #         }
                #         data[subject_id][step][sections[section_id]]['data'].append(new_data)

        # except:
        #     print('error')
        #     data.pop(subject_id)


print(data.keys())
pickle.dump(obj=data, file=open('raw_data_all', 'wb'))


#todo: bugs - 43
#todo: time to update data - when bitwin start and srop..