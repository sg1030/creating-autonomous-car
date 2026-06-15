import numpy as np

def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta

def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle

def interpolate(traj, min_dis):
    new_list=[]
    p=0
    for i in range(len(traj)):
        j = (i+1)%len(traj)
        dis_x= traj[j,0]-traj[i,0]
        dis_y= traj[j,1]-traj[i,1]
        dis=np.sqrt(dis_x**2+dis_y**2)
        if dis >min_dis:
            for k in range (int(dis//min_dis)):
                new_point=[traj[i,0]+dis_x*(k+1)/(dis//min_dis+1),traj[i,1]+dis_y*(k+1)/(dis//min_dis+1)]
                new_index = i+k+p+1
                new_list.append([new_point,new_index])
                # new_list.append([[wall[i,0]+dis_x*(k+1)/(dis//min_dis+1),wall[i,1]+dis_y*(k+1)/(dis//min_dis+1)],i+k+p+1])
            p+=int(dis//min_dis)

    refined_traj=insert_new_points(traj, new_list)

    return refined_traj

def insert_new_points(array , new_list):
    new_value = np.zeros(array.shape[1])
    for i in range(len(new_list)):
        new_value[:2] = new_list[i][0]
        if array.shape[1] > 2:
            new_value[2:] = array[new_list[i][1]-1,2:]
        array = np.insert(array,new_list[i][1],new_value ,axis=0)
    return array