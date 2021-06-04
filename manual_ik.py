import math as mt
from geometry_msgs.msg import Pose
from tf.transformations import *
import numpy as np

def ur5e_ik(pose):
    # ur5e parameter
    d1 = 0.163
    a2 = -0.425
    a3 = -0.392
    d4 = 0.134
    d5 = 0.1
    d6 = 0.1

    rot_mat = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    rot_mat[0][3] = pose.position.x
    rot_mat[1][3] = pose.position.y
    rot_mat[2][3] = pose.position.z
    (nx, ox, ax, px) = (rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], rot_mat[0][3])
    (ny, oy, ay, py) = (rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], rot_mat[1][3])
    (nz, oz, az, pz) = (rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], rot_mat[2][3])


    # solve theta 1
    m = d6*ay - py
    n = ax*d6 - px
    temp = mt.sqrt(m**2 + n**2 - d4**2)
    theta1 = np.array([mt.atan2(m, n) - mt.atan2(d4, temp),
                       mt.atan2(m, n) - mt.atan2(d4, -temp)])

    # solve theta 5
    temp = np.arccos(ax*np.sin(theta1) - ay*np.cos(theta1))
    theta5 = np.hstack((temp, -temp))

    # solve theta 6
    mm = nx*np.sin(theta1) - ny*np.cos(theta1)
    nn = ox*np.sin(theta1) - oy*np.cos(theta1)
    theta6 = np.arctan2(mm, nn)

