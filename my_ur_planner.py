#!/usr/bin/env python2
from quadrillion_test import *
from moveit_test import MoveGroupPythonInteface
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
import copy
from enum import Enum
from ur5eIKFast import ur5e_ik_fast


ControlMode = Enum('Mode', ('my_ur_ik', 'moveit', 'ikfast'))


def way_points_by_interpolating(start_pose, goal_pose):
    # this function is used for position interpolating and orientation is assumed to be unchanged
    default_interval = 0.05
    interval_x = goal_pose.position.x - start_pose.position.x
    interval_y = goal_pose.position.y - start_pose.position.y
    interval_z = goal_pose.position.z - start_pose.position.z
    interval = [abs(interval_x), abs(interval_y), abs(interval_z)]
    interval.sort(reverse=True)
    largest_interval = interval[0]

    way_points = [copy.deepcopy(start_pose)]
    interval_number = int(largest_interval / default_interval)
    for i in range(interval_number):
        interpolating_pose = Pose()
        interpolating_pose.position.x = start_pose.position.x + i * (interval_x / interval_number)
        interpolating_pose.position.y = start_pose.position.y + i * (interval_y / interval_number)
        interpolating_pose.position.z = start_pose.position.z + i * (interval_z / interval_number)
        interpolating_pose.orientation = start_pose.orientation
        way_points.append(copy.deepcopy(interpolating_pose))

    way_points.append(goal_pose)
    return way_points


class RobotStateMonitor:

    def __init__(self):
        self.joint_point = JointTrajectoryPoint()

    def update_robot_state(self, data):
        self.joint_point = data.actual


class MyRobotPlanner(object):

    def __init__(self, topic_command='/arm_controller/command', topic_state='/arm_controller/state', control_mode=ControlMode.ikfast):
        self.robot = MoveGroupPythonInteface()
        self.pub = rospy.Publisher(topic_command, JointTrajectory, queue_size=10)
        self.pub_map = rospy.Publisher('joint1_state', Float64, queue_size=10)
        self.robot_monitor = RobotStateMonitor()
        self.control_mode = control_mode
        rospy.Subscriber(topic_state, JointTrajectoryControllerState, self.robot_monitor.update_robot_state)
        print('robot go home')
        self.robot.go_home()

    # in first situation gaol point is a joint value
    # in the other two situations goal point is a pose
    # this defers because the rospy.sub in main func is different
    def control_robot(self, goal_point):
        if self.control_mode == ControlMode.my_ur_ik:
            self.control_robot_without_moveit(goal_point)
        elif self.control_mode == ControlMode.moveit:
            self.control_robot_with_moveit(goal_point)
        else:
            self.control_using_ikfast(goal_point)

    def control_robot_without_moveit(self, goal_point):
        start_point = self.robot_monitor.joint_point
        complete_point(start_point)
        start_point.time_from_start = rospy.Duration.from_sec(0)
        complete_point(goal_point)
        goal_point.positions = nearer_position(start_point.positions, goal_point.positions)
        # print(goal_point)
        traj, _ = traj_generate_with_two_points(start_point, goal_point)
        self.pub.publish(traj)
        self.pub_map.publish(self.robot_monitor.joint_point.positions[0])

    def control_robot_with_moveit(self, goal_point):
        default_duration = rospy.Duration.from_sec(0.1)
        print("get a catch point, the position is ", goal_point.pose)
        start_pose = self.robot.group.get_current_pose()
        # way_points = way_points_by_interpolating(start_pose.pose, goal_point.pose)
        # (plan, fraction) = self.robot.group.compute_cartesian_path(way_points, 0.01, 0.0)
        self.robot.group.set_pose_target(goal_point.pose)
        plan = self.robot.group.plan()
        time_to_goal = goal_point.header.stamp - rospy.Time.now()
        print("try to catch it in ", time_to_goal.to_sec())
        if time_to_goal.to_sec() < 0:
            time_to_goal = default_duration
        self.modify_time(plan.joint_trajectory.points, time_to_goal.to_sec())
        print("excute, points are", plan.joint_trajectory.points)
        self.robot.group.execute(plan, wait=False)
        # old version
        # start_point = self.robot_monitor.joint_point
        # complete_point(start_point)
        # start_point.time_from_start = rospy.Duration.from_sec(0)
        # complete_point(goal_point)
        # goal_point.positions = nearer_position(start_point.positions, goal_point.positions)
        # self.robot.group.set_joint_value_target(goal_point.positions)
        # plan = self.robot.group.plan()
        # self.modify_time(plan.joint_trajectory.points)
        # self.pub.publish(plan.joint_trajectory)

    def control_using_ikfast(self, goal_pose, velocity=()):
        velocity = (0, 0, -1)
        default_duration = rospy.Duration.from_sec(0.1)
        print("get a catch point, the position is ", goal_pose.pose)
        start_point = self.robot_monitor.joint_point
        complete_point(start_point)
        start_point.time_from_start = rospy.Duration.from_sec(0)
        goal_point_ik_joint_space = ur5e_ik_fast(goal_pose.pose)
        # print("solutions are")
        # for solution in goal_point_ik_joint_space:
        #     print(solution)
        if not goal_point_ik_joint_space:
            print("out of range")
            return
        best_solution = best_ik_solution(start_point.positions, goal_point_ik_joint_space)
        goal_point = JointTrajectoryPoint()
        goal_point.positions = best_solution
        complete_point(goal_point)
        time_to_goal = goal_pose.header.stamp - rospy.Time.now()
        print("try to catch it in ", time_to_goal.to_sec())
        # print("best solution is", best_solution)
        if time_to_goal.to_sec() < 0:
            time_to_goal = default_duration
        goal_point.time_from_start = 0.7 * time_to_goal
        # print(goal_point)
        traj, _ = traj_generate_with_two_points(start_point, goal_point)
        if velocity:
            back_pose = goal_pose
            velocity_norm = (velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2) ** 0.5
            back_pose.pose.position.x = 0.1 * velocity[0] / velocity_norm
            back_pose.pose.position.y = 0.1 * velocity[1] / velocity_norm
            back_pose.pose.position.z = 0.1 * velocity[2] / velocity_norm
            back_point_ik_solutions = ur5e_ik_fast(back_pose.pose)
            if back_point_ik_solutions:
                best_back_point_ik_solution = best_ik_solution(goal_point.positions, back_point_ik_solutions)
                back_point = JointTrajectoryPoint()
                back_point.positions = best_back_point_ik_solution
                back_point.time_from_start = rospy.Duration.from_sec(0.2) + goal_point.time_from_start
                traj.points.append(back_point)

        self.pub.publish(traj)
        self.pub_map.publish(self.robot_monitor.joint_point.positions[0])

    def modify_time(self, points, expected_time):
        if not points:
            print("empty points")
            exit(1)
        previous_last_time = points[-1].time_from_start.to_sec()
        for i in range(len(points)):
            new_time = expected_time*points[i].time_from_start.to_sec()/previous_last_time
            points[i].time_from_start = rospy.Time.from_sec(new_time)
            points[i].velocities = []
            points[i].accelerations = []


def complete_point(point):
    if not len(point.positions):
        point.positions = [0, 0, 0, 0, 0, 0]
    if not len(point.velocities):
        point.velocities = [0, 0, 0, 0, 0, 0]
    if not len(point.accelerations):
        point.accelerations = [0, 0, 0, 0, 0, 0]


def nearer_position(start_position, goal_position):
    result_position = []
    equivalent_position = lambda x: x - 2 * math.pi if x > 0 else x + 2 * math.pi
    nearer_position_between_two_points = lambda x, a, b: a if abs(x - a) < abs(x - b) else b
    for (spi, gpi) in zip(start_position, goal_position):
        result_position.append(nearer_position_between_two_points(spi, gpi, equivalent_position(gpi)))

    return result_position


def best_ik_solution(start_position, ik_solutions, return_distance=False):
    weight = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    distance = []
    if not ik_solutions:
        print('no ik solutions input')
    for ind in range(len(ik_solutions)):
        ik_solutions[ind] = nearer_position(start_position, ik_solutions[ind])
        distance.append(np.dot(weight,
                               np.abs(np.array(start_position) - np.array(ik_solutions[ind]))))
    # for ik_solution in ik_solutions:
    #     #nearer_ik_solution = nearer_position(start_position, ik_solution)
    #     # print("nearer_ik_solution")
    #     # print(nearer_ik_solution)
    #     distance.append(np.dot(weight,
    #                            np.abs(np.array(start_position) - np.array(ik_solution))))

    min_dis_ind = distance.index(min(distance))
    if return_distance:
        return distance[min_dis_ind]
    return ik_solutions[min_dis_ind]


if __name__ == '__main__':
    rospy.init_node('my_controller', anonymous=True, disable_signals=True)
    simulation = True
    if simulation:
        topic_command = '/arm_controller/command'
        topic_state = '/arm_controller/state'
    else:
        topic_command = '/scaled_pos_traj_controller/command'
        topic_state = '/scaled_pos_traj_controller/state'

    control_mode = ControlMode.ikfast
    my_robot_planner = MyRobotPlanner(topic_command=topic_command, topic_state=topic_state, control_mode=control_mode)
    if control_mode == ControlMode.my_ur_ik:
        rospy.Subscriber('my_command', JointTrajectoryPoint, my_robot_planner.control_robot)
    else:
        rospy.Subscriber('catch_point', PoseStamped, my_robot_planner.control_robot)

    rospy.spin()

