#!/usr/bin/python
# -*- coding: UTF-8 -*-
from quadrillion_test import *
from control_msgs.msg import JointTrajectoryControllerState
from moveit_test import MoveGroupPythonInteface
import moveit_msgs.msg
import geometry_msgs.msg


class RobotStateMonitor:

    def __init__(self):
        self.joint1_pos = []
        self.joint1_vec = []
        self.joint1_acc = []
        self.ti = []
        self.record = False

    def update_robot_state(self, data):
        if not self.record:
            pass
        self.ti.append(data.header.stamp.to_sec())
        if len(data.actual.positions) == 0:
            self.joint1_pos.append(0)
        else:
            self.joint1_pos.append(data.actual.positions[0])
        if len(data.actual.velocities) == 0:
            self.joint1_vec.append(0)
        else:
            self.joint1_vec.append(data.actual.velocities[0])
        if len(data.actual.accelerations) == 0:
            self.joint1_acc.append(0)
        else:
            self.joint1_acc.append(data.actual.accelerations[0])

    def plot(self):
        if self.record:
            print("error, can't plot while recording")
            exit(1)
        plt.plot(self.ti, self.joint1_pos, label='position')
        plt.plot(self.ti, self.joint1_vec, label='velocity')
        plt.plot(self.ti, self.joint1_acc, label='acceleration')
        plt.legend()
        plt.show()


def moveit_test():
    robot = MoveGroupPythonInteface()
    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
    joint1_monitor = RobotStateMonitor()
    rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, joint1_monitor.update_robot_state)
    robot.go_home()

    start_point = JointTrajectoryPoint()
    start_point.positions = [0, 0, 0, 0, 0, 0]
    start_point.velocities = [0, 0, 0, 0, 0, 0]
    start_point.accelerations = [0, 0, 0, 0, 0, 0]
    start_point.time_from_start = rospy.Duration.from_sec(0)
    goal_point = JointTrajectoryPoint()
    goal_point.positions = [-0.4, -1.8, 1.8, 0, 1.2, 3.1]
    goal_point.velocities = [0, 0, 0, 0, 0, 0]
    goal_point.accelerations = [0, 0, 0, 0, 0, 0]
    goal_point.time_from_start = rospy.Duration.from_sec(1)
    my_traj1, para_list1 = traj_generate_with_two_points(start_point, goal_point)
    print("publish first trajectory")
    joint1_monitor.record = True
    pub.publish(my_traj1)

    time.sleep(1.5)
    ti = start_point.time_from_start.to_sec() + 0.5
    start_point = point_interpolation_fifth(para_list1, ti)
    start_point.time_from_start = rospy.Duration.from_sec(0)
    goal_point.positions = [-0.8, -1.8, 1.8, 0, 1.2, 3.1]
    goal_point.time_from_start = rospy.Duration.from_sec(0.5)
    my_traj2, para_list2 = traj_generate_with_two_points(start_point, goal_point)
    print("publish second trajectory")
    pub.publish(my_traj2)
    time.sleep(0.5)
    joint1_monitor.record = False
    joint1_monitor.plot()


def real_robot_test():
    robot = MoveGroupPythonInteface()
    pub = rospy.Publisher('/scaled_pos_traj_controller/command', JointTrajectory, queue_size=10)  # check this
    joint1_monitor = RobotStateMonitor()
    rospy.Subscriber('/scaled_pos_traj_controller/state', JointTrajectoryControllerState,
                     joint1_monitor.update_robot_state)  # check this
    robot.go_home()
    time.sleep(3)

    start_point = JointTrajectoryPoint()
    # joint_goal = robot.get_current_joint_values()
    # if type(joint_goal) is not list:
    #     print('your code is wrong')
    #     exit(1)
    start_point.positions = [-0.4, -1.8, 1.8, 0, 1.2, 3.1]
    start_point.velocities = [0, 0, 0, 0, 0, 0]
    start_point.accelerations = [0, 0, 0, 0, 0, 0]
    start_point.time_from_start = rospy.Duration.from_sec(0)
    goal_point = JointTrajectoryPoint()
    goal_point.positions = [0.4, -1.8, 1.8, 0, 1.2, 3.1]
    goal_point.velocities = [0, 0, 0, 0, 0, 0]
    goal_point.accelerations = [0, 0, 0, 0, 0, 0]
    goal_point.time_from_start = rospy.Duration.from_sec(1)
    my_traj1, para_list1 = traj_generate_with_two_points(start_point, goal_point)
    print("publish first trajectory")
    joint1_monitor.record = True
    pub.publish(my_traj1)

    time.sleep(0.5)
    ti = start_point.time_from_start.to_sec() + 0.5
    start_point = point_interpolation_fifth(para_list1, ti)
    start_point.time_from_start = rospy.Duration.from_sec(0)
    goal_point.positions = [0.8, -1.8, 1.8, 0, 1.2, 3.1]
    goal_point.time_from_start = rospy.Duration.from_sec(0.5)
    my_traj2, para_list2 = traj_generate_with_two_points(start_point, goal_point)
    print("publish second trajectory")
    pub.publish(my_traj2)
    time.sleep(0.5)
    joint1_monitor.record = False
    joint1_monitor.plot()


if __name__ == '__main__':
    #real_robot_test()
    moveit_test()
