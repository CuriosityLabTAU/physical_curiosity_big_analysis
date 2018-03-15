# from to_nao_robot_error
# --> get list of section['robot'][i]-section['to_nao'][i]


# from section['robot'][i] and mirror matrix --> x
mirror_matrix=np.eye(8)
mirror_matrix[0, 0] = 0
mirror_matrix[4, 4] = 0
mirror_matrix[0, 4] = 1
mirror_matrix[4, 0] = 1
mirror_matrix[1, 1] = 0
mirror_matrix[5, 5] = 0
mirror_matrix[1, 5] = -1
mirror_matrix[5, 1] = -1
mirror_matrix = mirror_matrix[(0, 1, 4, 5),]
mirror_matrix = mirror_matrix[:, (0, 1, 4, 5)]





# from all x=skeleton_vectors and all section['to_nao'][i]=robot_vectors --> get matrix baseline
skeleton_vectors = np.vstack((skeleton_vectors, section['skeleton'][i][(0, 1, 4, 5),]))
robot_vectors = np.vstack((robot_vectors, section['robot'][i][(0, 1, 4, 5),]))
pinv_skeleton = pinv(skeleton_vectors, 0.00001)
# Amat = np.dot(pinv_skeleton, robot_vectors)
Amat = np.dot(robot_vectors.T, pinv_skeleton.T)



# ==> in file matrix_error: change mirror_matrix to baseline_matrix