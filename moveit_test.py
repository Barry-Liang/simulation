
#!/usr/bin/env python2

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

## BEGIN_SUB_TUTORIAL imports
##
## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. More on these below. We also import `rospy`_ and some messages that we will use:
##

import sys
import rospy
import time
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory
from moveit_commander.conversions import pose_to_list
## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupPythonInteface(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonInteface, self).__init__()

    ## BEGIN_SUB_TUTORIAL setup
    ##
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    if rospy.is_shutdown():
        rospy.init_node('move_group_python_interface',
                        anonymous=True)
    else:
        print("initialized in other place")

    ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
    ## the robot:
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
    ## to the world surrounding the robot:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to one group of joints.  In this case the group is the joints in the Panda
    ## arm so we set ``group_name = panda_arm``. If you are using a different robot,
    ## you should change this value to the name of your robot arm planning group.
    ## This interface can be used to plan and execute motions on the Panda:
    group_name = "manipulator"
    group = moveit_commander.MoveGroupCommander(group_name)

    ## We create a `DisplayTrajectory`_ publisher which is used later to publish
    ## trajectories for RViz to visualize:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    ## END_SUB_TUTORIAL

    ## BEGIN_SUB_TUTORIAL basic_info
    ##
    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()
    #print "============ Reference frame: %s" % planning_frame

    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    #print "============ End effector: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Robot Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    # print "============ Printing robot state"
    # print robot.get_current_state()
    #print ""
    ## END_SUB_TUTORIAL

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.group = group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names

  def go_home(self):
      group = self.group
      joint_goal = group.get_current_joint_values()
      joint_goal[0] = 1.427560859726321
      joint_goal[1] = -2.936315136835836
      joint_goal[2] = 1.769998306768569
      joint_goal[3] = -1.975275823522433
      joint_goal[4] = 0.143235467068575
      joint_goal[5] = 0
      group.go(joint_goal, wait=True)
      print ("finished")
      print (self.group.get_current_pose().pose)
      group.stop()
      current_joints = self.group.get_current_joint_values()
      return all_close(joint_goal, current_joints, 0.01)

  def go_to_pose(self, pose):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    #test1 = rospy.get_time()
    #print ("test1 = ")
    #print (test1)
    group = self.group

    pose_goal = geometry_msgs.msg.Pose()
    group.set_pose_target(pose)

    ## Now, we call the planner to compute the plan and execute it.
    group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    print ("reach")
    group.stop()
    #test2 = rospy.get_time()
    #print ("test2 = ")
    #print (test2)
    #print (test2 - test1)
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()

    ## END_SUB_TUTORIAL

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = self.group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)


def main():
    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
    robot = MoveGroupPythonInteface()
    robot.go_home()
    pose1 = geometry_msgs.msg.Pose()
    pose2 = geometry_msgs.msg.Pose()
    pose3 = geometry_msgs.msg.Pose()
    pose1.orientation.x = 0
    pose1.orientation.y = 0
    pose1.orientation.z = 0
    pose1.orientation.w = 1
    pose1.position.x = 0.4
    pose1.position.y = 0.4
    pose1.position.z = 0.4

    pose2.orientation.x = 0
    pose2.orientation.y = 0
    pose2.orientation.z = 0
    pose2.orientation.w = 1
    pose2.position.x = -0.4
    pose2.position.y = 0.4
    pose2.position.z = 0.4

    pose3.orientation.x = 0
    pose3.orientation.y = 0
    pose3.orientation.z = 0
    pose3.orientation.w = 1
    pose3.position.x = -0.4
    pose3.position.y = 0.4
    pose3.position.z = 0.7

    print('go to pose1')

    robot.go_to_pose(pose1)

    robot.group.set_pose_target(pose2)
    robot.group.set_goal_orientation_tolerance
    traj1 = robot.group.plan()

    pub.publish(traj1.joint_trajectory)
    print("start moving")

    # time.sleep(time_to_finish/2)

    # pub.publish(traj2.joint_trajectory)

    # robot.group.execute(traj2, wait=True)


    # time.sleep(5)

    # robot.group.set_pose_target(pose1)
    # robot.group.set_goal_orientation_tolerance
    # traj3 = robot.group.plan()
    # pub.publish(traj3.joint_trajectory)

def real_robot_test():
    pub = rospy.Publisher('/pos_joint_traj_controller/command',JointTrajectory, queue_size=10)
    robot = MoveGroupPythonInteface()
    #robot.go_home()
    pose_goal = geometry_msgs.msg.Pose()
    pose1 = geometry_msgs.msg.Pose()
    pose2 = geometry_msgs.msg.Pose()
    pose3 = geometry_msgs.msg.Pose()
    pose1.orientation.x = 0
    pose1.orientation.y = 0
    pose1.orientation.z = 0
    pose1.orientation.w = 1
    pose1.position.x = 0.4
    pose1.position.y = 0.4
    pose1.position.z = 0.4

    pose2.orientation.x = 0
    pose2.orientation.y = 0
    pose2.orientation.z = 0
    pose2.orientation.w = 1
    pose2.position.x = 0
    pose2.position.y = 0.4
    pose2.position.z = 0.4

    pose3.orientation.x = 0
    pose3.orientation.y = 0
    pose3.orientation.z = 0
    pose3.orientation.w = 1
    pose3.position.x = 0
    pose3.position.y = 0.4
    pose3.position.z = 0.5

    print('robot go to pose1')

    robot.go_to_pose(pose1)

    robot.group.set_pose_target(pose2)
    robot.group.set_goal_orientation_tolerance
    traj1 = robot.group.plan()
    time_to_finish = traj1.joint_trajectory.points[-1].time_from_start.nsecs/(1e9)
    

    robot.group.set_pose_target(pose3)
    robot.group.set_goal_orientation_tolerance
    traj2 = robot.group.plan()

    pub.publish(traj1.joint_trajectory)
    print("start moving")
    time.sleep(time_to_finish/2)

    pub.publish(traj2.joint_trajectory)

    robot.group.execute(traj2, wait=True)


    time.sleep(5)

    robot.group.set_pose_target(pose1)
    robot.group.set_goal_orientation_tolerance
    traj3 = robot.group.plan()
    pub.publish(traj3.joint_trajectory)





if __name__ == '__main__':
    main()