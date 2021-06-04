#!/usr/bin/python

import rospy
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
#from ring_predict.srv import *
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import tf
from geometry_msgs.msg import PoseStamped


def get_pre_param(position_set, time_series): ## no assumption of g #input position array, return x = a0 + a1*t y = a2 + a3*t z = a4 + a5*t + a6*t*t
    time = time_series.reshape(-1, 1)
    positionx = position_set[:, 0].reshape(-1, 1)
    positiony = position_set[:, 1].reshape(-1, 1)
    positionz = position_set[:, 2].reshape(-1, 1)
    modelx = LinearRegression(fit_intercept=True)
    modely = LinearRegression(fit_intercept=True)
    modelz = LinearRegression(fit_intercept=True)

    second_time = np.power(time, 2)
    #zero_z = np.ones((len(positionz), 1))

    modelx.fit(time, positionx)
    modely.fit(time, positiony)
    modelz.fit(np.hstack((time, second_time)), positionz)

    a0 = modelx.intercept_[0]
    a1 = modelx.coef_[0][0]
    a2 = modely.intercept_[0]
    a3 = modely.coef_[0][0]
    a4 = modelz.intercept_[0]
    a5 = modelz.coef_[0][0]
    a6 = modelz.coef_[0][1]

    return a0, a1, a2, a3, a4, a5, a6

def get_pre_param2(position_set, time_series): ## assume g=9.8 #input position array, return x = a0 + a1*t y = a2 + a3*t z = a4 + a5*t + a6*t*t
    time = time_series.reshape(-1, 1)
    positionx = position_set[:, 0].reshape(-1, 1)
    positiony = position_set[:, 1].reshape(-1, 1)
    positionz = position_set[:, 2].reshape(-1, 1)
    modelx = LinearRegression(fit_intercept=True)
    modely = LinearRegression(fit_intercept=True)
    modelz = LinearRegression(fit_intercept=True)

    second_time = np.power(time, 2) * (-4.9)
    #zero_z = np.ones((len(positionz), 1))

    modelx.fit(time, positionx)
    modely.fit(time, positiony)
    modelz.fit(time, positionz - second_time)

    a0 = modelx.intercept_[0]
    a1 = modelx.coef_[0][0]
    a2 = modely.intercept_[0]
    a3 = modely.coef_[0][0]
    a4 = modelz.intercept_[0]
    a5 = modelz.coef_[0][0]
    a6 = -4.9

    return a0, a1, a2, a3, a4, a5, a6

def solve_second_order(a, b, c): #solve second order equation for real solution or return -1
    delta = b * b - 4 * a * c
    if delta < 0:
        print ('cannot solve delta < 0')
        return -1
    else:
        x1 = (-b + math.sqrt(delta))/(2*a)
        x2 = (-b - math.sqrt(delta)) / (2 * a)

        if a > 0:
            return x1, x2
        else:
            return x2, x1

def solve_time_max(theta): #time when z>0
    a = theta[6]
    b = theta[5]
    c = theta[4]
    t = solve_second_order(a=a, b=b, c=c)
    if t!=-1:
        tmax = t[0]
        #tmax = math.floor(t[0])
    else:
        print('error, no tmax found not to the ground')
        return -1
    return tmax

def cal_quartic_ik(args_list):
    a, b, c, d, e = args_list

    D = 3 * pow(b, 2) - 8 * a * c
    E = -pow(b, 3) + 4 * a * b * c - 8 * pow(a, 2) * d
    F = 3 * pow(b, 4) + 16 * pow(a, 2) * pow(c, 2) - 16 * a * pow(b, 2) * c + 16 * pow(a, 2) * b * d - 64 * pow(a,
                                                                                                                3) * e

    A = D ** 2 - 3 * F
    B = D * F - 9 * pow(E, 2)
    C = F ** 2 - 3 * D * pow(E, 2)

    delta = B ** 2 - 4 * A * C

    if (D == 0) & (E == 0) & (F == 0):
        """ 4 real root"""
        x = -b / (4 * a)
        return 1, [x]
    if (A == 0) & (B == 0) & (C == 0) & (D * E * F != 0):
        """ 2 real 1 double imaginary"""
        x1 = (-b * D + 9 * E) / (4 * a * D)
        x234 = (-b * D - 3 * E) / (4 * a * D)
        return 2, [x1, x234]
    if (E == 0) & (F == 0) & (D != 0):
        """ 2 double"""
        if D > 0:  # real
            x13 = (-b + math.sqrt(D)) / (4 * a)
            x24 = (-b - math.sqrt(D)) / (4 * a)
            return 2, [x13, x24]

        if D < 0:  # imaginary
            # x13 = (-b + cmath.sqrt(D))/(4*a)
            # x24 = (-b - cmath.sqrt(D)) / (4 * a)
            return 0, 0
    if (A * B * C != 0) & (delta == 0):
        """ 2 double real """
        x3 = (-b - np.sign(A * B * E) * math.sqrt(D - B / A)) / (4 * a)
        x4 = (-b - np.sign(A * B * E) * math.sqrt(D - B / A)) / (4 * a)
        if A * B > 0:  # rest inequal roots
            x1 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) + math.sqrt(2 * B / A)) / (4 * a)
            x2 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) - math.sqrt(2 * B / A)) / (4 * a)
            return 4, [x1, x2, x3, x4]
        if A * B < 0:  # rest two imaginary
            # x1 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) + cmath.sqrt(2 * B / A)) / (4 * a)
            # x2 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) - cmath.sqrt(2 * B / A)) / (4 * a)
            return 2, [x3, x4]
    if delta > 0:
        """" two unequal two imaginary"""
        z1 = A * D + 3 * ((-B + math.sqrt(delta)) / 2.0)
        z2 = A * D + 3 * ((-B - math.sqrt(delta)) / 2.0)

        # print """ z1 =  """, z1
        # print """ z2 =  """, z2
        # print """ abs(z1) =  """, abs(z1)
        # print """ abs(z2) =  """, abs(z2)

        z = D ** 2 - D * (np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) + \
            (np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) ** 2 - 3 * A

        x1 = (-b + np.sign(E) * math.sqrt(
            (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
              + math.sqrt((2 * D - np.sign(z1) * pow(abs(z1), 1.0 / 3.0) - np.sign(z2) * pow(abs(z2), 1.0 / 3.0)
                           + 2 * math.sqrt(z)) / 3.0)) / (4 * a)
        x2 = (-b + np.sign(E) * math.sqrt(
            (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
              - math.sqrt((2 * D - np.sign(z1) * pow(abs(z1), 1.0 / 3.0) - np.sign(z2) * pow(abs(z2), 1.0 / 3.0)
                           + 2 * math.sqrt(z)) / 3.0)) / (4 * a)

        # ingnore imaginray
        return 2, [x1, x2]
    if delta < 0:
        if E == 0:
            if (D > 0) & (F > 0):
                """ four inequal real """
                x1 = (-b + math.sqrt(D + 2 * math.sqrt(F))) / (4 * a)
                x2 = (-b - math.sqrt(D + 2 * math.sqrt(F))) / (4 * a)
                x3 = (-b + math.sqrt(D - 2 * math.sqrt(F))) / (4 * a)
                x4 = (-b - math.sqrt(D - 2 * math.sqrt(F))) / (4 * a)
                return 4, [x1, x2, x3, x4]
            else:
                """ two pairs imaginary """
                # ignore imaginary
                print (" two pairs imaginary ")
                return 0, 0
        else:
            if (D > 0) & (F > 0):
                """ four inequal real """
                theta = math.acos((3 * B - 2 * A * D) / (2 * A * math.sqrt(A)))
                y1 = (D - 2 * math.sqrt(A) * math.cos(theta / 3.0)) / 3.0
                y2 = (D + math.sqrt(A) * (math.cos(theta / 3.0) + math.sqrt(3) * math.sin(theta / 3.0))) / 3.0
                y3 = (D + math.sqrt(A) * (math.cos(theta / 3.0) - math.sqrt(3) * math.sin(theta / 3.0))) / 3.0

                x1 = (-b + np.sign(E) * math.sqrt(y1) + (math.sqrt(y2) + math.sqrt(y3))) / (4 * a)
                x2 = (-b + np.sign(E) * math.sqrt(y1) - (math.sqrt(y2) + math.sqrt(y3))) / (4 * a)
                x3 = (-b - np.sign(E) * math.sqrt(y1) + (math.sqrt(y2) - math.sqrt(y3))) / (4 * a)
                x4 = (-b - np.sign(E) * math.sqrt(y1) - (math.sqrt(y2) - math.sqrt(y3))) / (4 * a)

                return 4, [x1, x2, x3, x4]
            else:
                """ two pairs imaginary """
                # ignore imaginary
                print (" two pairs imaginary ")
                return 0, 0

def solve_time_start(theta, robot_loc, robot_range): #reach radius input theta robot_location, robot_reach return possible timeset
    a0 = theta[0]
    a1 = theta[1]
    a2 = theta[2]
    a3 = theta[3]
    a4 = theta[4]
    a5 = theta[5]
    a6 = theta[6]

    x = robot_loc[0]
    y = robot_loc[1]
    z = robot_loc[2]

    R = robot_range

    a = a6 * a6
    b = 2 * a6 * a5
    c = 2 * a6 * a4 - 2 * a6 * z + a5 * a5 + a3 * a3 + a1 * a1
    d = 2 * a4 * a5 - 2 * z * a5 + 2 * a2 * a3 - 2 * a3 * y + 2 * a1 * a0 - 2 * a1 * x
    e = (a4 - z) * (a4 - z) + (a2 - y) * (a2 - y) + (a0 - x) * (a0 - x) - R * R

    result = cal_quartic_ik(args_list=(a, b, c, d, e))

    if result[0] ==0:
        print ('imaginary root, no time start found')
        return []
    else:
        timeset = np.array(result[1])
        #timeset = np.ceil(timeset)
        #timeset = np.ceil(timeset)
        return timeset

def solve_time_period(theta, robot_loc, robot_range):   #### reach a height and a sphere radius
    timemax = solve_time_max(theta)
    # print (timemax)
    if timemax == -1:
        print('wrong timemax')
        return -1, -1
    interresult = solve_time_start(theta=theta, robot_loc=robot_loc, robot_range=robot_range)
    #print (interresult)
    if len(interresult) == 0:
        print ('no interception')
        return -1, -1
    timeset = np.hstack((interresult, timemax)).reshape(1, -1)
    timeset = timeset[np.where(timeset>0)]
    if len(timeset) < 2:
        print ('error')
        return -1, -1
    else:
        timeset.sort()
        t1 = timeset[0]
        t2 = timeset[1]
        return t1, t2

def solve_time_period2(theta, robot_loc, robot_range, zcatch):  ### reach certain height
    t = solve_second_order(theta[6], theta[5], theta[4] - zcatch)
    if t == -1:
        print ("trajectory too low")
        return -1
    else:
        position = time_to_loc(theta, t[0])
        #print (position)
        rangeflag = (position[0] - robot_loc[0])*(position[0] - robot_loc[0]) + (position[1] - robot_loc[1])*(position[1] - robot_loc[1]) + (position[2] - robot_loc[2])*(position[2] - robot_loc[2]) - robot_range* robot_range
        if rangeflag > 0:
            print ("out of range")
            return -1
        else:
            return t[0]

def solve_time_period3(theta, robot_loc, robot_range, catch_ratio=0.75):  ### reach certain radius with a ratio
    interresult = solve_time_start(theta=theta, robot_loc=robot_loc, robot_range=robot_range*catch_ratio)
    if len(interresult) == 0:
        print ('no interception')
        return -1
    timeset = interresult[np.where(interresult > 0)]
    if len(timeset) != 2:
        print ('error !=2')
        return -1
    else:
        if timeset[0] > timeset[1]:
            return timeset[1]
        else:
            return timeset[0]


def time_to_loc(theta, time):
    x = theta[0] + theta[1] * time
    y = theta[2] + theta[3] * time
    z = theta[4] + theta[5] * time + theta[6] * time * time
    position = (x, y, z)
    return position

def cal_velocity_vector(theta, time):
    v = (theta[1], theta[3], theta[5] + 2 *theta[6] * time)
    #v2 = (v[0].tolist()[0], v[1].tolist()[0], v[2])
    v2 = (v[0], v[1], v[2])
    norm = np.linalg.norm(v2, axis=0, keepdims=True)
    v = v2 / norm
    return v

def xyzwtoR(x, y, z, w):
    R = np.zeros((3,3))
    R[0][0] = 1 - 2*y*y - 2*z*z
    R[1][0] = 2*x*y + 2*w*z
    R[2][0] = 2*x*z - 2*w*y
    R[0][1] = 2*x*y - 2*w*z
    R[1][1] = 1 - 2*x*x - 2*z*z
    R[2][1] = 2*y*z + 2*w*x
    R[0][2] = 2*x*z + 2*w*y
    R[1][2] = 2*y*z - 2*w*x
    R[2][2] = 1 - 2*x*x -2*y*y
    return R

def solve_orientation_from_v(v):
    R1 = tf.transformations.quaternion_matrix((math.sin(math.pi / 4), 0.00000, math.sin(math.pi / 4), -0.00000))
    vector = np.array([v[0], v[1], v[2]]).reshape(-1, 1)
    vector = vector * (-1.0)
    print (vector)
    #vector = np.array([0.6, 0.000, 0.8]).reshape(-1, 1)
    #print (vector)
    vector = np.dot(np.linalg.inv(R1[:3,:3]), vector)
    angle1 = math.atan2(-vector[2], vector[0])
    angle2 = math.atan2(vector[1], math.sqrt(vector[0] * vector[0] + vector[2] * vector[2]))

    R2 = tf.transformations.rotation_matrix(angle1, (0, 1, 0), (0, 0, 0))
    R3 = tf.transformations.rotation_matrix(angle2, (0, 0, 1), (0, 0, 0))

    R4 = np.dot(np.dot(R1, R2), R3)
    result = tf.transformations.quaternion_from_matrix(R4) #x, y, z, w
    return result


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
    rospy.init_node('move_group_python_interface',
                    anonymous=True)

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
    # group_names = robot.get_group_names()
    # print ("============ Robot Groups:", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    # print ("============ Printing robot state")
    # print (robot.get_current_state())
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
      joint_goal[0] = 0
      joint_goal[1] = -pi / 2
      joint_goal[2] = 3*pi / 4
      joint_goal[3] = 5*pi / 4 #+ 0.643501110063
      joint_goal[4] = pi/2
      joint_goal[5] = 0
      group.go(joint_goal, wait=True)
      print ("finished")
      print (self.group.get_current_pose().pose)
      group.stop()
      current_joints = self.group.get_current_joint_values()
      return all_close(joint_goal, current_joints, 0.01)

  def go_to_pose(self, px, py, pz, ox, oy, oz, ow):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    #test1 = rospy.get_time()
    #print ("test1 = ")
    #print (test1)
    group = self.group

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = ox
    pose_goal.orientation.y = oy
    pose_goal.orientation.z = oz
    pose_goal.orientation.w = ow
    pose_goal.position.x = px
    pose_goal.position.y = py
    pose_goal.position.z = pz
    group.set_pose_target(pose_goal)

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

  def fast_go_to_pose(self, px, py, pz, ox, oy, oz, ow, ratio):
    group = self.group

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = ox
    pose_goal.orientation.y = oy
    pose_goal.orientation.z = oz
    pose_goal.orientation.w = ow
    pose_goal.position.x = px
    pose_goal.position.y = py
    pose_goal.position.z = pz
    group.set_pose_target(pose_goal)
    group.set_goal_orientation_tolerance


    traj = group.plan()

    new_traj = RobotTrajectory()
    new_traj = traj

    n_joints = len(traj.joint_trajectory.joint_names)
    n_points = len(traj.joint_trajectory.points)

    total_time = traj.joint_trajectory.points[n_points - 1].time_from_start


    spd = ratio

    points = list(traj.joint_trajectory.points)

    for i in range(n_points):
        point = JointTrajectoryPoint()
        point.time_from_start = traj.joint_trajectory.points[i].time_from_start / spd
        point.velocities = list(traj.joint_trajectory.points[i].velocities)
        point.accelerations = list(traj.joint_trajectory.points[i].accelerations)
        point.positions = traj.joint_trajectory.points[i].positions

        for j in range(n_joints):
            point.velocities[j] = point.velocities[j] * spd
            point.accelerations[j] = point.accelerations[j] * spd * spd

        points[i] = point

    new_traj.joint_trajectory.points = points

    group.execute(new_traj, wait=True)


def Poses_to_nparray(Posesarray):
    array = []
    for i in range(len(Posesarray)):
        array.append(pose_to_list(Posesarray[i])) #px py pz ox oy oz ow
    nparray = np.array(array)
    return nparray

def Times_to_nparray(Timepoints):
    array = []
    headtime = Timepoints[0].stamp.to_sec()
    for i in range(len(Timepoints)):
        t = Timepoints[i].stamp.to_sec() - headtime
        array.append(t)
    nparray = np.array(array).squeeze()
    return nparray

def Pre_control():
    try:
        #RobotGroup = MoveGroupPythonInteface()

        # RobotGroup.fast_go_to_pose(px=0.35996741, py=0.2061016,
        #                            pz=0.37996408, ox=0.89459152979471523, oy=0.0001, oz=0.44729576455252823,
        #                            ow=0.00011064464941967898, ratio=3.0)
        #RobotGroup.go_home()

        robot_location = (0.5, 0.0, 0.0)    #ji de gai
        robot_reach = 1.0
        rospy.init_node('planner',anonymous=True)
        pub = rospy.Publisher('catch_point', PoseStamped, queue_size=10)
        rate = rospy.Rate(50)
        goal_pose = PoseStamped()
        result_location = []
        seq = 1
        # RobotGroup.fast_go_to_pose(px=0.35996741, py=0.2061016,
        #                       pz=0.37996408, ox=math.cos(math.pi / 4), oy=0.0,
        #                       oz=math.cos(math.pi / 4), ow=0.0,ratio = 1.0)

        # RobotGroup.fast_go_to_pose(px=0.35996741, py=0.2061016,
        #                            pz=0.37996408, ox=0.894413867027, oy=-1.5591861136e-05, oz=0.447240235467, ow=7.74954373525e-05, ratio=3.0)
        print ("============ Press `Enter` to continue")
        raw_input()
        # RobotGroup.fast_go_to_pose(px=0.35996741, py=0.2061016,
        #                            pz=0.37996408, ox=0.89442719,
        #                            oy=0.0, oz=0.4472136, ow=0.0, ratio=3.0)

        while True:#rospy.is_shutdown():
            rospy.wait_for_service('Ring_Information')
            get_camera_result = rospy.ServiceProxy('Ring_Information', ring_Info)
            result = get_camera_result(0)
            print (len(result.Ringposes))

            if len(result.Ringposes) > 20:

                PoseSet = Poses_to_nparray(result.Ringposes)
                TimeSet = Times_to_nparray(result.Timepoints)
                theta = get_pre_param2(PoseSet[:, :3], TimeSet)
                print (theta)
                np.save('/home/liangxiao/theta.npy', np.array(theta))
                # t1 ,t2 = solve_time_period(theta=theta, robot_loc=robot_location, robot_range=robot_reach)
                t1 = solve_time_period2(theta=theta, robot_loc=robot_location, robot_range=robot_reach, zcatch=0.60)
                t2 = solve_time_period3(theta=theta, robot_loc=robot_location, robot_range=robot_reach, catch_ratio=1.0)
                if t1 == -1 or t2 == -1:
                    print ('cannot catch')
                else:
                    if seq<=200:
                        # print ("t1 = %s, t2 = %s" %(t1,t2))
                        # tcatch = (t1 + t2)/2
                        print (t1, t2)
                        tcatch = (t1 + t2)/2
                        print ("tcatch = %s" % tcatch)
                        catch_position = time_to_loc(theta, tcatch)
                        result_location.append(catch_position)
                        catch_orientation = solve_orientation_from_v(cal_velocity_vector(theta, tcatch))
                    #
                        print ("Planed catching location = (%s, %s, %s)" %catch_position)
                        deltax = -0.2
                        deltay = -0.0
                        goal_pose.pose.position.x = catch_position[0] - 0.5 + deltax#rviz transformation
                        goal_pose.pose.position.y = catch_position[1] + deltay
                        goal_pose.pose.position.z = catch_position[2]

                        goal_pose.pose.orientation.x = catch_orientation[0]
                        goal_pose.pose.orientation.y = catch_orientation[1] #-math.sin(math.pi/4)
                        goal_pose.pose.orientation.z = catch_orientation[2] #0.0
                        goal_pose.pose.orientation.w = catch_orientation[3] #math.cos(math.pi/4)

                        goal_pose.header.seq = seq
                        goal_pose.header.stamp = rospy.Time.from_sec(tcatch+result.Timepoints[0].stamp.to_sec())
                        # print ("xiaoge time is:")
                        # print (result.Timepoints[0].stamp.to_sec())
                        goal_pose.header.frame_id = 'my_planner'

                        pub.publish(goal_pose)
                        seq = seq + 1

                #     print (catch_orientation)
                #     #RobotGroup.fast_go_to_pose(px=catch_position[0].tolist()[0], py=catch_position[1].tolist()[0], pz=catch_position[2].tolist()[0], ox=math.cos(math.pi/4), oy=0.0, oz=math.cos(math.pi/4), ow=0.0, ratio=5.0)
                #     RobotGroup.fast_go_to_pose(px=catch_position[0].tolist()[0], py=catch_position[1].tolist()[0],
                #                            pz=catch_position[2].tolist()[0], ox=catch_orientation[0], oy=catch_orientation[1],
                #                            oz=catch_orientation[2], ow=catch_orientation[3], ratio=5.0)

                np.save('/home/liangxiao/camera_result.npy', PoseSet)
                np.save('/home/liangxiao/time_result.npy', TimeSet)
                xxx = np.array(result_location)
                print (xxx.shape)
                np.save('/home/liangxiao/location_result.npy', xxx)
            else:
                print ('No data received')

            # RobotGroup.go_to_pose(px=0.35996741, py=0.2061016,
            #                       pz=0.37996408, ox=0, oy=0, oz=0, ow=1)

            #rate.sleep()




    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    Pre_control()
