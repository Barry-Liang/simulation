import subprocess
from geometry_msgs.msg import Pose
from tf.transformations import *
import math
import random
import matplotlib.pyplot as plt


def ur5e_ik_fast(pose):
    rot_mat = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    rot_mat[0][3] = pose.position.x
    rot_mat[1][3] = pose.position.y
    rot_mat[2][3] = pose.position.z
    ur5e_ik_fast_dir = '/home/liangxiao/catkin_ws/src/Ring_Simulation/ur5eIK'
    cmd = [ur5e_ik_fast_dir]
    for i in range(3):
        for j in range(4):
            cmd.append(str(rot_mat[i][j]))

    try:
        out_bytes = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        #print(out_bytes)
        return []

    out_text = out_bytes.decode('utf-8')
    # print(out_text)
    out_lines = out_text.split('\n')
    result = []
    for ind in range(1, len(out_lines)-1):
        data_in_line = out_lines[ind].partition(':')[-1]
        data_split = data_in_line.split(',')
        data_split.pop()
        result_ind = []
        for data in data_split:
            result_ind.append(float(data.strip()))

        result.append(result_ind)

    return result


def draw_workspace():
    pose1 = Pose()
    pose1.orientation.x = 0
    pose1.orientation.y = -math.sin(math.pi / 8)
    pose1.orientation.z = 0
    pose1.orientation.w = math.cos(math.pi / 8)
    pose1.position.z = 0.7
    xlist = []
    ylist = []
    for i in range(1000):
        pose1.position.x = random.random() - 0.5
        pose1.position.y = random.random() - 0.5
        result = ur5e_ik_fast(pose1)
        if result:
            xlist.append(pose1.position.x)
            ylist.append(pose1.position.y)

    plt.scatter(xlist, ylist)
    plt.show()


if __name__ == '__main__':
    pose1 = Pose()
    pose1.orientation.x = 0
    pose1.orientation.y = 0
    pose1.orientation.z = 0
    pose1.orientation.w = 1
    pose1.position.x = -0.2
    pose1.position.y = -0.2
    pose1.position.z = 0.7

    print("for pose")
    print(pose1)
    ik_solutions = ur5e_ik_fast(pose1)




