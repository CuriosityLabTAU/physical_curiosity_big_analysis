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

affdex_list=['emotions','expressions','measurements','face_points','face_x_alignment',
             'face_y_alignment','face_tracking_deltax','face_distance','face_detected','tega_affect_msg','attention_detected']

mypath = '/home/matan/Desktop/new_data/'

files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.bag' in f]
subjects_with_no_angles_topic=[]
data = {}
# for each bag file
for f in files:
    info = f.split('_')
    subject_id = float(info[4])

    if subject_id==1065:
        continue

    print subject_id

    if subject_id > 0.0:
        print('processing ', subject_id)
        count+=1
        print "count= ",count
        data[subject_id] = {}
        data[subject_id][0] = {}


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


        #is_nao_angles_topic
        is_nao_angles_topic=False

        is_working= False

        current_skeleton_angle = None
        current_nao_movements = None
        current_nao_command = None

        first_time = True

        # try:

        for topic, msg, t in bag.read_messages():
            # get the first time, for reference
            if first_time:
                t0 = t
                first_time = False

            if 'log' in topic:
                if'current_state' in msg.data:
                    step=int(msg.data.split(': ')[1])
                    if step in data[subject_id]:
                        continue

                    data[subject_id][step]={}
                    if step == 1:
                        section_id = 1
                        print 'here'
                    else:
                        section_id = 0

                if 'matrix' in msg.data:
                    if step in data[subject_id]:
                        matrix=msg.data.split(':')[1]
                        data[subject_id][step]['matrix'] = np.array(np.matrix([[float(x)for x in row if x!=''] for row in [line[2:-1].split(' ') for line in  matrix[2:-1].split('\n ')]]))


                if'task' in msg.data:
                    task_full_name=msg.data.split(',')[2]
                    task=task_full_name.split('/')[-1]


            if 'flow' in topic:
                #   get the transformation

                if msg.data.isdigit():
                    if step in data[subject_id]:
                        data[subject_id][step]['transformation'] = int(msg.data)

                  # parse the time to sections (learning, task1, task2, task3)
                if 'start' in msg.data:
                    if step in data[subject_id]:
                        is_working=True
                        data[subject_id][step][sections[section_id]] = {
                            'start': t,
                            'stop': None,
                            'data': [],
                            'task':task
                        }

                if 'stop' in msg.data:
                    if step in data[subject_id]:
                        if sections[section_id] in data[subject_id][step]:
                            is_working=False
                            data[subject_id][step][sections[section_id]]['stop'] = t
                            task = 0
                            section_id += 1

                            if step == 12:
                                bag.close()
                                break

                if section_id >= len(sections):
                    section_id=0


            if 'affdex' in topic:
                # print msg
                dict = {}
                for m in affdex_list:
                    dict[m] = eval('msg.' + m)
                    if m == 'face_points':
                        n_list = []
                        o_list = dict[m]
                        for point in o_list:
                            n_list.append({"x": point.x, "y": point.y})
                        dict[m] = n_list
                affdex = dict


              # for each section
            # if 'nao_movements' in topic:
            #    #          get command to robot
            #     current_nao_movements = msg.data


            # time between skeleton_angles is +- 0.04
            # time between robot angles is +- 0.09


            if 'nao_angles_topic' in topic:
                #       get command to robot

                is_nao_angles_topic = True

                list_of_movements_str = msg.data
                list_of_movements = list_of_movements_str[1:-1].split(', ')

                current_nao_movements = (list_of_movements[2:4]+[0.0,0.0]+list_of_movements[-6:-4]+[0.0,0.0])

                current_nao_movements= str([float(x) for x in current_nao_movements])
                current_nao_movements= current_nao_movements[1:-1]

            if 'to_nao' in topic:
                #       get command to robot
                if'change_pose' in msg.data:
                    current_nao_command = msg.data

            if 'skeleton_angle' in topic:
                #       get raw skeleton markers
                current_skeleton_angle = msg.data
                # print current_skeleton_angle

                    #dict[id] = dict[section] = array(dict{skeleton, robot, time})
                if is_working==True:
                    if step in data[subject_id]:
                        if sections[section_id] in data[subject_id][step]:
                            if current_skeleton_angle is not None and current_nao_movements is not None and current_nao_command is not None:
                                new_data = {
                                    'time': (t - t0).to_sec(),
                                    'skeleton': current_skeleton_angle,
                                    'robot_cimmand':current_nao_command,
                                    'robot': current_nao_movements,
                                    'affdex':affdex
                                }
                                data[subject_id][step][sections[section_id]]['data'].append(new_data)

        if is_nao_angles_topic==False:
            subjects_with_no_angles_topic.append(subject_id)

        # except:
        #     print('error')
        #     data.pop(subject_id)


print(data.keys())
print 'subjects_with_no_angles_topic: ',subjects_with_no_angles_topic

pickle.dump(obj=data, file=open('data/raw_data_10_10', 'wb'))

data[subject_id][step][sections[section_id]]['data']= {
                                    'time': (t - t0).to_sec(),
                                    'skeleton': current_skeleton_angle,
                                    'robot_cimmand': current_nao_command,
                                    'robot': current_nao_movements,
                                    'affdex': affdex}