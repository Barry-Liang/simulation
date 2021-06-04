#!/usr/bin/env python2

from Pre_n_move import *
from my_ur_planner import *
import sympy as sp
import sys
from tf.transformations import quaternion_matrix

def pre_part():
    global seq , tcatch, theta, min_distance, last_length
    goal_pose = PoseStamped()
    rospy.wait_for_service('Ring_Information')
    get_camera_result = rospy.ServiceProxy('Ring_Information', ring_Info)
    result = get_camera_result(0)
    print (len(result.Ringposes))
    if len(result.Ringposes)<last_length:
        min_distance=sys.maxint
        print("New trun is starting.")
    else:
        last_length=len(result.Ringposes)


    if len(result.Ringposes) > 20:

        PoseSet = Poses_to_nparray(result.Ringposes)
        TimeSet = Times_to_nparray(result.Timepoints)
        theta = get_pre_param2(PoseSet[:, :3], TimeSet)
        #print (theta)
        np.save('/home/liangxiao/theta.npy', np.array(theta))
        # t1 ,t2 = solve_time_period(theta=theta, robot_loc=robot_location, robot_range=robot_reach)
        t1 = solve_time_period2(theta=theta, robot_loc=robot_location, robot_range=robot_reach, zcatch=0.60)
        t2 = solve_time_period3(theta=theta, robot_loc=robot_location, robot_range=robot_reach, catch_ratio=0.9)
        if t1 == -1 or t2 == -1 or (t1 < t2):
            print ('cannot catch')
            return
        else:
            tcatch,distance = catch_point_least_cartesian_distance(t1, t2, theta)
            #tcatch,distance = catch_point_least_joint_pace_distance(t2, t1, theta)
            if distance<min_distance:
                min_distance=distance
                #tcatch = catch_point_least_joint_pace_distance(t2, t1, theta)
                print('tcatch is ', tcatch)

                # tcatch = (t1 + t2) / 2
                # print ("tcatch = %s" % tcatch)
                catch_position = time_to_loc(theta, tcatch)
                result_location.append(catch_position)
                catch_orientation = solve_orientation_from_v(cal_velocity_vector(theta, tcatch))
                #
                # print ("Planed catching location = (%s, %s, %s)" % catch_position)
                rotation_matrix = quaternion_matrix(catch_orientation)

                deltax = -0.0
                deltay = -0.0
                goal_pose.pose.position.x = catch_position[0] - 0.13*rotation_matrix[0, 2] - 0.5 + deltax  # rviz transformation
                goal_pose.pose.position.y = catch_position[1] - 0.13*rotation_matrix[1, 2] + deltay
                goal_pose.pose.position.z = catch_position[2] - 0.13*rotation_matrix[2, 2]

                goal_pose.pose.orientation.x = catch_orientation[0]
                goal_pose.pose.orientation.y = catch_orientation[1]  # -math.sin(math.pi/4)
                goal_pose.pose.orientation.z = catch_orientation[2]  # 0.0
                goal_pose.pose.orientation.w = catch_orientation[3]  # math.cos(math.pi/4)

                goal_pose.header.seq = seq
                goal_pose.header.stamp = rospy.Time.from_sec(tcatch + result.Timepoints[0].stamp.to_sec())

                goal_pose.header.frame_id = 'my_planner'

                seq = seq + 1
                np.save('/home/liangxiao/camera_result.npy', PoseSet)
                np.save('/home/liangxiao/time_result.npy', TimeSet)
                xxx = np.array(result_location)
                # print (xxx.shape)
                np.save('/home/liangxiao/location_result.npy', xxx)
                velocity = cal_velocity_vector(theta, tcatch)
                return goal_pose, velocity
            else:
                print ('Distance is not decreasing')
                np.save('/home/liangxiao/camera_result.npy', PoseSet)
                np.save('/home/liangxiao/time_result.npy', TimeSet)
                xxx = np.array(result_location)
                # print (xxx.shape)
                np.save('/home/liangxiao/location_result.npy', xxx)
                return



    else:
        print ('No data received')


def distance_with_ee(t, theta,position):
    # current_pose = my_robot_planner.robot.group.get_current_pose()
    # x = current_pose.pose.position.x
    # y = current_pose.pose.position.y
    # z = current_pose.pose.position.z
    x =position[0]
    y = position[1]
    z = position[2]
    t_position = time_to_loc(theta, t)
    return (t_position[0] - x)**2 + (t_position[1] - y)**2 + (t_position[2] - z)**2


def get_nearest_position_time(theta, t_start, t_end, position):
    # current_pose = my_robot_planner.robot.group.get_current_pose()
    # x = current_pose.pose.position.x
    # y = current_pose.pose.position.y
    # z = current_pose.pose.position.z
    x = position[0]
    y = position[1]
    z = position[2]
    a = 48.2
    b = 14.7*theta[5]
    c = theta[5]**2 + 9.8*theta[4] + theta[1]**2 + theta[3]**2 - 9.8*z
    d = theta[0]*theta[1] +theta[2]*theta[3] + theta[4]*theta[5] - theta[5]*z -theta[3]*y - theta[1]*x

    t = sp.Symbol('t')
    f = a*t ** 3 + b*t**2 + c*t + d
    t = sp.solve(f)
    result = []
    for ti in t:
        if isinstance(ti, (int, long, float)):
            if (ti > t_start) and (ti < t_end):
                result.append(ti)

    return result


def catch_point_least_cartesian_distance(t1, t2, theta, position=[0.7, 0.0, 0.0]):
    distance = []
    t_least_distance = get_nearest_position_time(theta, t2, t1, position)
    t_least_distance.append(t1)
    t_least_distance.append(t2)
    for ti in t_least_distance:
        distance.append(distance_with_ee(ti, theta,position))

    least_index = distance.index(min(distance))
    tcatch = t_least_distance[least_index]
    return tcatch, min(distance)


def catch_point_least_joint_pace_distance(t1, t2, theta):
    # sampling
    t_catch = -1
    min_distance = -1
    sample_num=10
    for i in range(sample_num+1):
        time_i = t1 + i*(t2-t1)/(sample_num+1)
        pose = Pose()
        position = time_to_loc(theta, time_i)  # (x, y, z)
        velocity = cal_velocity_vector(theta, time_i)
        orientation = solve_orientation_from_v(velocity)  # (x, y, z, w)
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        joint_values = ur5e_ik_fast(pose)
        if not joint_values:
            continue
        distance = best_ik_solution(my_robot_planner.robot_monitor.joint_point.positions,
                                       joint_values, return_distance=True)
        if (distance < min_distance) or (min_distance == -1):
            min_distance = distance
            t_catch = time_i
            print("get one", t_catch)
    # this funct has solved the best joint space value
    # return time result calculation
    # but this can let the code keep the same logic
    return t_catch,min_distance


if __name__ == '__main__':

    # initialization
    robot_location = (0.505, 0.0, 0.0)  # pay attention to this setting!!!
    robot_reach = 1.0
    rospy.init_node('planner', anonymous=True)

    result_location = []
    seq = 1

    tcatch=0
    theta=[]
    min_distance=sys.maxint
    last_length=0

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
    print ("============ Press `Enter` to continue")
    raw_input()

    while True:
        catch_pose_stamped, velocity = pre_part()
        if catch_pose_stamped:
            print("get a catch pose")
            my_robot_planner.control_using_ikfast(catch_pose_stamped, velocity=velocity)
