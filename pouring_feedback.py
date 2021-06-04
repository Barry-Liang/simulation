import math
import time

from geometry_msgs.msg import Pose, PoseStamped
from my_ur_planner import MyRobotPlanner, ControlMode
import rospy


class FeedbackPouringControl:

    # robot_initial_position: pose; geometry_parameter: [h, r]
    def __init__(self, robot, robot_init_position, geometry_parameter):
        self.robot_init_position = robot_init_position.position
        self.robot_init_orientation = robot_init_position.orientation
        self.max_height = geometry_parameter[0]
        self.cup_radius = geometry_parameter[1]
        self.angle = math.pi/6
        self.robot = robot

    def ratio_diff2height(self, ratio_diff):
        return 0.5*self.max_height*(ratio_diff + 1)

    def height2pose(self, height):
        expected_pose = Pose()
        expected_pose.orientation = self.robot_init_orientation
        expected_pose.position.x = self.robot_init_position.x
        expected_pose.position.y = self.robot_init_position.y + height*math.sin(self.angle) + \
                                   self.cup_radius*math.cos(self.angle)
        expected_pose.position.z = self.robot_init_position.z - height*math.sin(self.angle) + \
                                   self.cup_radius*math.sin(self.angle)
        return expected_pose

    def feedback_control(self, ratio):
        expected_ratio = 0.5
        ratio_diff = ratio - expected_ratio
        height = self.ratio_diff2height(ratio_diff)
        pose = self.height2pose(height)
        goal_pose = PoseStamped()
        goal_pose.header.seq = 1
        goal_pose.header.stamp = rospy.Time.from_sec(0.1)
        goal_pose.header.frame_id = 'my_planner'
        self.robot.control_using_ikfast(goal_pose)


if __name__ == '__main__':
    simulation_flag = False
    if simulation_flag:
        topic_command = '/arm_controller/command'
        topic_state = '/arm_controller/state'
    else:
        topic_command = '/scaled_pos_traj_controller/command'
        topic_state = '/scaled_pos_traj_controller/state'
    control_mode = ControlMode.ikfast
    my_robot_planner = MyRobotPlanner(topic_command=topic_command,
                                      topic_state=topic_state, control_mode=control_mode)
    pouring_control = FeedbackPouringControl(my_robot_planner)
    print ("robot ready to receive command")

    for ratio in range(0, 5):
        ratio /= 10.0
        pouring_control.feedback_control(ratio)
        time.sleep(3)

