import rospy, math, time
import numpy as np
import matplotlib.pyplot as plt
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def point_interpolation_fifth(para_list, ti):
    point = JointTrajectoryPoint()
    for jointNo in range(len(para_list)):
        pos_i = np.dot(
            np.array([1, ti, math.pow(ti, 2), math.pow(ti, 3), math.pow(ti, 4), math.pow(ti, 5)]),
            np.array(para_list[jointNo])
        )
        vec_i = np.dot(
            np.array([0, 1, 2 * ti, 3 * math.pow(ti, 2), 4 * math.pow(ti, 3), 5 * math.pow(ti, 4)]),
            np.array(para_list[jointNo])
        )
        acc_i = np.dot(
            [0, 0, 2, 6 * ti, 12 * math.pow(ti, 2), 20 * math.pow(ti, 3)],
            np.array(para_list[jointNo])
        )
        point.positions.append(pos_i)
        point.velocities.append(vec_i)
        point.accelerations.append(acc_i)

    point.time_from_start = rospy.Duration.from_sec(ti)
    return point


def traj_generate_with_two_points(joint_point_start, joint_point_goal, deltaT=0.2, interpolation=True):
    # joint_point_start, joint_point_goal = jointPoint
    # initialize trajectory message
    # interpolation = False
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = "world"  # replace this

    joint_name_list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                       'wrist_3_joint']
    traj.joint_names = joint_name_list

    # append point
    # calculate parameters for each joint
    para_list = []
    time_start = joint_point_start.time_from_start.to_sec()
    time_goal = joint_point_goal.time_from_start.to_sec()
    if not interpolation:
        traj.points.append(joint_point_goal)
        para_list = []
        return traj, para_list
    for i in range(len(joint_name_list)):
        A = np.mat([
            [1, time_start, math.pow(time_start, 2), math.pow(time_start, 3), math.pow(time_start, 4),
             math.pow(time_start, 5)],
            [0, 1, 2 * time_start, 3 * math.pow(time_start, 2), 4 * math.pow(time_start, 3),
             5 * math.pow(time_start, 4)],
            [0, 0, 2, 6 * time_start, 12 * math.pow(time_start, 2), 20 * math.pow(time_start, 3)],
            [1, time_goal, math.pow(time_goal, 2), math.pow(time_goal, 3), math.pow(time_goal, 4),
             math.pow(time_goal, 5)],
            [0, 1, 2 * time_goal, 3 * math.pow(time_goal, 2), 4 * math.pow(time_goal, 3), 5 * math.pow(time_goal, 4)],
            [0, 0, 2, 6 * time_goal, 12 * math.pow(time_goal, 2), 20 * math.pow(time_goal, 3)]
        ])
        b = np.array(
            [joint_point_start.positions[i], joint_point_start.velocities[i], joint_point_start.accelerations[i],
             joint_point_goal.positions[i], joint_point_goal.velocities[i], joint_point_goal.accelerations[i]])
        x = np.linalg.solve(A, b)
        para_list.append(x)

    # append points
    for i in range(int(math.ceil((time_goal - time_start) / deltaT))):
        if not i:
            continue
        time_i = time_start + i * deltaT
        point_i = point_interpolation_fifth(para_list, time_i)
        # for jointNo in range(len(para_list)):
        #     pos_i = np.dot(
        #         np.array([1, time_i, math.pow(time_i, 2), math.pow(time_i, 3), math.pow(time_i, 4), math.pow(time_i, 5)]),
        #         np.array(para_list[jointNo])
        #     )
        #     vec_i = np.dot(
        #         np.array([0, 1, 2 * time_i, 3 * math.pow(time_i, 2), 4 * math.pow(time_i, 3), 5 * math.pow(time_i, 4)]),
        #         np.array(para_list[jointNo])
        #     )
        #     acc_i = np.dot(
        #         [0, 0, 2, 6 * time_i, 12 * math.pow(time_i, 2), 20 * math.pow(time_i, 3)],
        #         np.array(para_list[jointNo])
        #     )
        #     point_i.positions.append(pos_i)
        #     point_i.velocities.append(vec_i)
        #     point_i.accelerations.append(acc_i)
        #
        # point_i.time_from_start = rospy.Duration.from_sec(time_i)
        traj.points.append(point_i)

    traj.points.append(joint_point_goal)

    return traj, para_list


if __name__ == '__main__':
	pass
    # rospy.init_node('quadrillion_test', anonymous=True)
    # pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
    # jointGroupPoints1 = []
    # jointGroupPoints2 = []
    # for i in range(6):
    #     jointIPoint1 = jointPoint()
    #     jointIPoint2 = jointPoint()
    #     jointIPoint2.pos = math.pi / 4
    #     jointIPoint2.vec = 0
    #     jointIPoint2.acc = 0
    #     jointIPoint2.time = 1
    #     jointGroupPoints1.append(jointIPoint1)
    #     jointGroupPoints2.append(jointIPoint2)
	#
    # traj = trajGenerateByTwoPoints(jointGroupPoints1, jointGroupPoints2)
    # print('publish trajectory to the topic')
    # # print(traj)
    # pub.publish(traj)
# pos = []
# vec = []
# acc = []
# ti = []
# for point in traj.points:
# 	pos.append(point.positions[0])
# 	vec.append(point.velocities[0])
# 	acc.append(point.accelerations[0])
# 	ti.append(point.time_from_start.to_sec())

# plt.plot(ti, pos, label = 'position')
# plt.plot(ti, vec, label = 'velocity')
# plt.plot(ti, acc, label = 'acceleration')
# plt.legend()
# plt.show()
