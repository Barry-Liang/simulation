#!/usr/bin/python
from ring_pre_catch_mini import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class UrRobot(MyRobotPlanner):

    def __init__(self, robot_location):
        topic_command = '/arm_controller/command'
        topic_state = '/arm_controller/state'
        control_mode = ControlMode.ikfast
        super(UrRobot, self).__init__(topic_command=topic_command, topic_state=topic_state, control_mode=control_mode)
        #self.robot_init_state = [0, -2.936315136835836, 1.769998306768569, -1.975275823522433, 0.143235467068575, 0]
        #self.robot.group.go(self.robot_init_state)
        self.robot_init_state = [1.427560859726321, -2.936315136835836, 1.769998306768569, -1.975275823522433, 0.143235467068575, 0]
        self.traj_para = []   #### joint path parameters
        self.robot_location = robot_location

    def get_current_position(self, t):  ###end-effector position in world coordinates
        current_point = JointTrajectoryPoint()
        current_point.positions = self.robot_init_state
        if self.traj_para:
            current_point = point_interpolation_fifth(self.traj_para, t)
            self.robot.group.go(current_point.positions, wait=True)
           # print('moved', current_point)
        position = self.robot.group.get_current_pose().pose.position
        return [position.x + self.robot_location[0], position.y + self.robot_location[1], position.z + self.robot_location[2]], current_point

    def updata_trajectory(self, traj_para):
        self.traj_para = traj_para



def main():
    rospy.init_node('my_controller', anonymous=True, disable_signals=True)
    ################ load data
    dir_str = '/home/liangxiao/Documents/12-24-data/'
    camera_data = np.load(dir_str + 'camera_result-hr1.npy')
    time_data = np.load(dir_str + 'time_result-hr1.npy')
    time_data=time_data.reshape(-1,1)
    print(camera_data.shape)
    print(time_data.shape)

    ##########  simulation data
    dt=0.001
    a0, a1, b0, b1, c0, c1, fac_x, fac_y, fac_z = -2.2, 2.5, 0, -0.3, 1.3, 5.5,0.004, 0.004, 0.008
    A, alpha, beta, gama, omega =  0.98081794 ,  0.7690529  ,  2.7736743  , -1.04374361 , 31.93226724
    orientation_param = [A, alpha, beta, gama, omega]
    time_sim = np.arange(0.0, 0.8, dt).reshape(-1,1)
    X_sim = a0 + a1 * time_sim + (np.random.randn(len(time_sim)) - 0.5).reshape(-1,1) * fac_x
    Y_sim = b0 + b1 * time_sim + (np.random.randn(len(time_sim)) - 0.5).reshape(-1,1) * fac_y
    Z_sim = c0 + c1 * time_sim - 4.9 * time_sim ** 2 + (np.random.randn(len(time_sim)) - 0.5).reshape(-1,1) * fac_z
    nx_sim = A * np.cos(alpha) * np.cos(beta) - np.sqrt(1 - A ** 2) * np.sin(beta) * np.cos(
        omega * time_sim + gama).reshape(-1,1) - np.sqrt(1 - A ** 2) * np.sin(alpha) * np.cos(beta) * np.sin(omega * time_sim + gama).reshape(-1,1)
    ny_sim = A * np.cos(alpha) * np.sin(beta) + np.sqrt(1 - A ** 2) * np.cos(beta) * np.cos(
        omega * time_sim + gama).reshape(-1,1) - np.sqrt(1 - A ** 2) * np.sin(alpha) * np.sin(beta) * np.sin(omega * time_sim + gama).reshape(-1,1)
    nz_sim = A * np.sin(alpha) + np.sqrt(1 - A ** 2) * np.cos(alpha) * np.sin(omega * time_sim + gama).reshape(-1,1)
    # nx_sim[np.where(nz_sim > 0)] = -nx_sim[np.where(nz_sim > 0)]
    # ny_sim[np.where(nz_sim > 0)] = -ny_sim[np.where(nz_sim > 0)]
    # nz_sim[np.where(nz_sim > 0)] = -nz_sim[np.where(nz_sim > 0)]
    X_sim = X_sim[200:]
    Y_sim = Y_sim[200:]
    Z_sim = Z_sim[200:]
    nx_sim = nx_sim[200:]
    ny_sim = ny_sim[200:]
    nz_sim = nz_sim[200:]
    T_sim = time_sim[200:]
    camera_data=np.hstack((X_sim,Y_sim,Z_sim,nx_sim,ny_sim,nz_sim,T_sim))
    time_data=T_sim[:]
    print(camera_data.shape)
    print(time_data.shape)
    ######## set robot parameter
    robot_location = (0.7, 0.0, 0.0)
    robot_reach = 1.0

    # init_orientation = robot.robot.group.get_current_pose().pose.orientation
    def draw_end_effector_joint_path():
        ############ use first i data to calculate catch point
        step=25
        fig = plt.figure()
        ax = Axes3D(fig)
        fig2 = plt.figure()
        ax1=fig2.add_subplot(321)

        ax2=fig2.add_subplot(322)

        ax3=fig2.add_subplot(323)

        ax4=fig2.add_subplot(324)

        ax5=fig2.add_subplot(325)

        ax6=fig2.add_subplot(326)
        fig2.tight_layout()
        fig2.subplots_adjust(left=0.1,right=0.7,top=0.9,bottom=0.1,wspace=0.4, hspace=0.6)
        for mode in ['ee','base','half']:
            robot = UrRobot(robot_location)
            catch_points_x = []
            catch_points_y = []
            catch_points_z = []
            robot_position_x = []
            robot_position_y = []
            robot_position_z = []
            joint0 = []
            joint1 = []
            joint2 = []
            joint3 = []
            joint4 = []
            joint5 = []
            joint_t = []
            success = 0
            tcatch = 0
            traj_para = []
            L=range(0, len(time_data)-1, step)
            if L[-1]!=len(time_data)-2:
                L.append(len(time_data)-2)
            for i in L:
                print('number of data=---------------------------')
                print(i)
                theta = get_pre_param2(camera_data[:i+1, :3], time_data[:i+1])  # fitting,
                # theta is the parameter of fitting in-flgiht trajectory
                t1 = solve_time_period2(theta=theta, robot_loc=robot_location, robot_range=robot_reach, zcatch=0.6)
                # intersection time with plane z=0.6
                t2 = solve_time_period3(theta=theta, robot_loc=robot_location, robot_range=robot_reach, catch_ratio=0.9)
                # intersection time with sphere r=0.9
                if t1 == -1 or t2 == -1 or (t1 < t2):
                    print ('cannot catch')
                else:
                    #### calculate catch point and save it
                    ####  present_time is executed time in last joint trajectory.
                    #### notice that each joint trajectory is finished at catching time,
                    #### so each update of catching time will interprut the path and only execute a part of it
                    if not i:
                        present_time = 0
                    else:
                        present_time = (time_data[i] - time_data[i - step])

                    position, joints_values = robot.get_current_position(present_time)
                    joints_position = joints_values.positions
                    joint_t.append(time_data[i])
                    joint0.append(joints_position[0])
                    joint1.append(joints_position[1])
                    joint2.append(joints_position[2])
                    joint3.append(joints_position[3])
                    joint4.append(joints_position[4])
                    joint5.append(joints_position[5])

                    robot_position_x.append(position[0])
                    robot_position_y.append(position[1])
                    robot_position_z.append(position[2])

                    ##### tcatch = solve_time_period2(theta=theta, robot_loc=robot_location, robot_range=robot_reach, zcatch=0.77)
                    if mode=='ee':
                        tcatch, _ = catch_point_least_cartesian_distance(t1, t2, theta, position)
                    #######  nearest point to end effector
                    if mode=='base':
                        tcatch, _ = catch_point_least_cartesian_distance(t1, t2, theta, robot_location)
                    #######  nearest point to base
                    if mode=='half':
                        tcatch = 0.5*(t1+t2)
                    #######   half point

                    # if catch_points_z:
                    #     tcatch, _ = catch_point_least_cartesian_distance(t1, t2, theta, [catch_points_x[-1], catch_points_y[-1],
                    #                                                                      catch_points_z[-1]])
                    # else:
                    #     tcatch, _ = catch_point_least_cartesian_distance(t1, t2, theta, position)

                    catch_point = time_to_loc(theta, tcatch)
                    v=cal_velocity_vector(theta, tcatch)
                    orientation=solve_orientation_from_v(v)
                    # calculate robot trajectory
                    start_point = joints_values
                    # start_point.positions = joints_position
                    complete_point(start_point)
                    start_point.time_from_start = rospy.Duration.from_sec(0)
                    goal_pose = Pose()
                    goal_pose.position.x = catch_point[0] - robot_location[0]
                    goal_pose.position.y = catch_point[1] - robot_location[1]
                    goal_pose.position.z = catch_point[2] - robot_location[2]
                    goal_pose.orientation.x = orientation[0]
                    goal_pose.orientation.y = orientation[1]
                    goal_pose.orientation.z = orientation[2]
                    goal_pose.orientation.w = orientation[3]
                    # print('goal pose is', goal_pose)
                    goal_point_ik_joint_space = ur5e_ik_fast(goal_pose)
                    # print("solutions are")
                    # for solution in goal_point_ik_joint_space:
                    #     print(solution)
                    if not goal_point_ik_joint_space:
                        print("out of range")
                        continue
                    best_solution = best_ik_solution(start_point.positions, goal_point_ik_joint_space)
                    goal_point = JointTrajectoryPoint()
                    goal_point.positions = best_solution
                    complete_point(goal_point)
                    goal_point.time_from_start = rospy.Duration.from_sec(tcatch - time_data[i])
                    _, traj_para = traj_generate_with_two_points(start_point, goal_point)
                    robot.updata_trajectory(traj_para)
                    catch_points_x.append(catch_point[0])
                    catch_points_y.append(catch_point[1])
                    catch_points_z.append(catch_point[2])
                    success += 1

            time_end = time_data[-1]
            time_step = step*time_end/len(time_data)
            time_catch_num = int(len(time_data)*(tcatch - time_end)/time_end)/step + 1
            time_left = [t*time_step for t in range(1, time_catch_num)]

            for t in time_left:
                position, joints_values = robot.get_current_position(t)
                joints_position = joints_values.positions
                joint_t.append(t + time_end)
                joint0.append(joints_position[0])
                joint1.append(joints_position[1])
                joint2.append(joints_position[2])
                joint3.append(joints_position[3])
                joint4.append(joints_position[4])
                joint5.append(joints_position[5])
                robot_position_x.append(position[0])
                robot_position_y.append(position[1])
                robot_position_z.append(position[2])
            tail_num=len(time_left)
            print("catching point num=======")
            print (success)
            if mode=='ee':
                catch_color='blue'
                main_color = 'purple'
                tail_color = 'purple'
            elif mode=='base':
                catch_color = 'green'
                main_color = 'brown'
                tail_color = 'brown'
            else:
                catch_color = 'red'
                main_color = 'gray'
                tail_color = 'gray'

            catch_points_x=np.array(catch_points_x)-robot_location[0]
            catch_points_y = np.array(catch_points_y) - robot_location[1]
            catch_points_z = np.array(catch_points_z) - robot_location[2]
            robot_position_x=np.array(robot_position_x)-robot_location[0]
            robot_position_y = np.array(robot_position_y) - robot_location[1]
            robot_position_z = np.array(robot_position_z)- robot_location[2]

            #ax.scatter(catch_points_x, catch_points_y, catch_points_z,marker='*',color='black',label='catching points')
            #ax.scatter(robot_position_x, robot_position_y, robot_position_z,marker='>',color='black',label='end-effector position')
            ax.plot3D(catch_points_x, catch_points_y, catch_points_z,marker='*',color=catch_color)
            ax.plot3D(robot_position_x[:len(joint_t)-tail_num+1], robot_position_y[:len(joint_t)-tail_num+1], robot_position_z[:len(joint_t)-tail_num+1],marker='>',linestyle='-',color=main_color)
            ax.plot3D(robot_position_x[len(joint_t)-tail_num:], robot_position_y[len(joint_t)-tail_num:], robot_position_z[len(joint_t)-tail_num:],marker='>',linestyle='--',color=tail_color)

            ax1.plot(joint_t[:len(joint_t)-tail_num+1], joint0[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax1.plot(joint_t[len(joint_t)-tail_num:], joint0[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

            ax2.plot(joint_t[:len(joint_t)-tail_num+1], joint1[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax2.plot(joint_t[len(joint_t)-tail_num:], joint1[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

            ax3.plot(joint_t[:len(joint_t)-tail_num+1], joint2[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax3.plot(joint_t[len(joint_t)-tail_num:], joint2[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

            ax4.plot(joint_t[:len(joint_t)-tail_num+1], joint3[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax4.plot(joint_t[len(joint_t)-tail_num:], joint3[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

            ax5.plot(joint_t[:len(joint_t)-tail_num+1], joint4[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax5.plot(joint_t[len(joint_t)-tail_num:], joint4[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

            ax6.plot(joint_t[:len(joint_t)-tail_num+1], joint5[:len(joint_t)-tail_num+1],linestyle='-',color=main_color)
            ax6.plot(joint_t[len(joint_t)-tail_num:], joint5[len(joint_t)-tail_num:],linestyle='--',color=tail_color)

        fig.legend(['method1:catching points','method1:ee path when ring is in camera view','method1:ee path when ring goes outside camera view',
                    'method2:catching points', 'method2:ee path when ring is in camera view',
                    'method2:ee path when ring goes outside camera view',
                    'method3:catching points', 'method3:ee path when ring is in camera view',
                    'method3:ee path when ring goes outside camera view'
                    ])
        fig2.legend(['method1:Inside camera view','method1:Outside camera view','method2:Inside camera view','method2:Outside camera view','method3:Inside camera view','method3:Outside camera view'],loc='right')
        fig2.tight_layout(rect=(0, 0, 1, 1))
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.set_zlabel('z/m')

        ax1.set_title('joint1')
        ax2.set_title('joint2')
        ax3.set_title('joint3')
        ax4.set_title('joint4')
        ax5.set_title('joint5')
        ax6.set_title('joint6')
        ax1.set_xlabel('time/s')
        ax1.set_ylabel('joint angle/rad')
        ax2.set_xlabel('time/s')
        ax2.set_ylabel('joint angle/rad')
        ax3.set_xlabel('time/s')
        ax3.set_ylabel('joint angle/rad')
        ax4.set_xlabel('time/s')
        ax4.set_ylabel('joint angle/rad')
        ax5.set_xlabel('time/s')
        ax5.set_ylabel('joint angle/rad')
        ax6.set_xlabel('time/s')
        ax6.set_ylabel('joint angle/rad')
        plt.show()

    def draw_trajectory(frequency=[1000,500,300,100,60, 30]):
        fig = plt.figure(1)
        ax1 = Axes3D(fig)
        fig2 = plt.figure(2)
        ax2=fig2.add_subplot(111)
        for f in frequency:
            L=range(0,time_data.shape[0]//5,1000/f)
            theta = get_pre_param2(camera_data[L, :3], time_data[L])  # fitting,
            # theta is the parameter of fitting in-flgiht trajectory
            x,y,z = time_to_loc(theta, time_data)
            ax1.plot3D(x[:,0],y[:,0],z[:,0],label="frequency"+str(f))
            # ax1.plot3D(camera_data[:, 0],camera_data[:, 1],camera_data[:, 2],label="real case")
            k = 3
            len_p=len(L)
            error_w_time = np.zeros((len_p - k, 3))
            error_all = np.zeros((len_p - k))
            pose=camera_data[L, :3]
            time=time_data[L,0]

            for i in range(k, len_p):
                theta = get_pre_param2(pose[:i, :3], time[:i])
                nihe = time_to_loc(theta, time_data[:,0])
                nihe = np.array(nihe).transpose()
                error_w_time[i - k, :] = np.sqrt(np.sum(np.power(nihe - camera_data[:, :3], 2), axis=0) / nihe.shape[0])
            error_all = np.sqrt(np.sum(np.power(error_w_time[:, :], 2), axis=1))
            # ax2.plot(time[k:], error_w_time[:, 0], label='x')
            # ax2.plot(time[k:], error_w_time[:, 1], label='y')
            # ax2.plot(time[k:], error_w_time[:, 2], label='z')
            ax2.plot(time[k:], error_all[:], label='Overall'+str(f))
            print(f)
            print(error_all[-1])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMSE of Predicted Trajectory (m)')
        ax1.legend()
        ax2.legend()
        plt.show()
    #draw_trajectory()

    draw_end_effector_joint_path()

def func2(x,p):
    A,a1,a2,a3,a4 = p
    theta = a4*x + a3
    item1 = np.cos(theta)
    item2 = np.sin(theta)
    item3 = np.ones(item1.shape)
    item4 = np.vstack([item3, item1, item2])
    item5 = np.array([[A*np.cos(a1)*np.cos(a2), -np.sqrt(1-A *A)*np.sin(a2), -np.sqrt(1-A *A)*np.sin(a1)*np.cos(a2) ],
                      [A*np.cos(a1)*np.sin(a2), +np.sqrt(1-A *A)*np.cos(a2), -np.sqrt(1-A *A)*np.sin(a1)*np.sin(a2)],
                      [A*np.sin(a1), 0, +np.sqrt(1-A * A)*np.cos(a1)]])
    item6 = np.dot(item5, item4)
    return item6.T

def residuals2(p,y,x):
    return (np.sum(np.power(y-func2(x, p),2), axis=1))

def test():
    dir_str = '/home/liangxiao/Documents/12-24-data/'
    camera_data = np.load(dir_str + 'camera_result-hr-perfect.npy')
    time_data = np.load(dir_str + 'time_result-hr-perfect.npy')
    fig = plt.figure()
    ax1 = Axes3D(fig)
    # ax1.plot(camera_data[:, 0], camera_data[:, 1], camera_data[:, 2],color='blue')




    a0, a1, b0, b1, c0, c1, fac_x, fac_y, fac_z = -2.2, 2.5, 0, -0.3, 1.3, 5.5, 0.002, 0.002, 0.005
    A,alpha,beta,gama,omega= -0.98081794,   0.7690529 ,   2.7736743 ,  -1.04374361,  31.93226724            #0.9,0,1,0,2*np.pi*4.9
    orientation_param = [A,alpha,beta,gama,omega]
    time_sim = np.arange(0.0, 0.8, 0.001)
    X_sim = a0 + a1 * time_sim + (np.random.randn(len(time_sim)) - 0.5) * fac_x
    Y_sim = b0 + b1 * time_sim + (np.random.randn(len(time_sim)) - 0.5) * fac_y
    Z_sim = c0 + c1 * time_sim - 4.9 * time_sim ** 2 + (np.random.randn(len(time_sim)) - 0.5) * fac_z
    nx_sim = A*np.cos(alpha)*np.cos(beta)-np.sqrt(1-A**2)*np.sin(beta)*np.cos(omega*time_sim+gama)-np.sqrt(1-A**2)*np.sin(alpha)*np.cos(beta)*np.sin(omega*time_sim+gama)
    ny_sim = A*np.cos(alpha)*np.sin(beta)+np.sqrt(1-A**2)*np.cos(beta)*np.cos(omega*time_sim+gama)-np.sqrt(1-A**2)*np.sin(alpha)*np.sin(beta)*np.sin(omega*time_sim+gama)
    nz_sim = A*np.sin(alpha)+np.sqrt(1-A**2)*np.cos(alpha)*np.sin(omega*time_sim+gama)


    # nx_sim[np.where(nz_sim>0)]= -nx_sim[np.where(nz_sim> 0)]
    # ny_sim[np.where(nz_sim> 0)] = -ny_sim[np.where(nz_sim> 0)]
    # nz_sim[np.where(nz_sim>0)] = -nz_sim[np.where(nz_sim >0)]
    #
    #
    # from scipy.signal import argrelextrema,savgol_filter
    # minimum=argrelextrema(nz_sim, np.greater)[0].tolist()
    # print(minimum)
    # for i in minimum:
    #     print(i)
    #     if abs(nz_sim[i])<0.05:
    #         nx_sim[i:] = -nx_sim[i:]
    #         ny_sim[i:] = -ny_sim[i:]
    #         nz_sim[i:] = -nz_sim[i:]
    # # nx_sim = savgol_filter(nx_sim, 31, 3, mode='nearest')
    # # ny_sim = savgol_filter(ny_sim, 31, 3, mode='nearest')
    # # nz_sim = savgol_filter(nz_sim, 31, 3, mode='nearest')
    #
    # def smooth(y, box_pts):
    #     box = np.ones(box_pts) / box_pts
    #     y_smooth = np.convolve(y, box, mode='same')
    #     return y_smooth
    #
    # nx_sim = smooth(nx_sim, 11)
    # ny_sim = smooth(ny_sim, 11)
    # nz_sim = smooth(nz_sim, 11)
    #
    # # from scipy.signal import butter, filtfilt
    # # # Filter requirements.
    # # fs = 30  # sample rate, Hz
    # # cutoff = 7.5  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    # # nyq = 0.5 * fs  # Nyquist Frequency
    # # order = 2  # sin wave can be approx represented as quadratic
    # #
    # # def butter_lowpass_filter(data, cutoff, fs, order):
    # #     normal_cutoff = cutoff / nyq
    # #     # Get the filter coefficients
    # #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # #     y = filtfilt(b, a, data)
    # #     return y
    # #
    # # print(nx_sim.shape[0])
    # # nx_sim=butter_lowpass_filter(nx_sim, cutoff, fs, order)
    # # ny_sim = butter_lowpass_filter(ny_sim, cutoff, fs, order)
    # # nz_sim = butter_lowpass_filter(nz_sim, cutoff, fs, order)
    # # print("after low pass filter")
    # # print(nx_sim.shape[0])
    #
    #




    X_sim = X_sim[200:]
    Y_sim = Y_sim[200:]
    Z_sim = Z_sim[200:]
    nx_sim= nx_sim[200:]
    ny_sim = ny_sim[200:]
    nz_sim = nz_sim[200:]
    T_sim = time_sim[200:]

    print(X_sim.shape)
    print(nx_sim.shape)
    ax1.plot(X_sim, Y_sim, Z_sim, color='red')
    ax1.quiver(X_sim[::25], Y_sim[::25], Z_sim[::25], nx_sim[::25], ny_sim[::25], nz_sim[::25],arrow_length_ratio=0.1,length=0.1,normalize=True,color='green')
    plt.axis("equal")
    fig2 = plt.figure()
    plt.scatter(T_sim,nx_sim)
    plt.scatter(T_sim, ny_sim)
    plt.scatter(T_sim, nz_sim)
    plt.legend(['x','y','z'])

    from scipy.optimize import leastsq,curve_fit
    nx_sim= nx_sim.reshape(-1,1)
    ny_sim = ny_sim.reshape(-1,1)
    nz_sim = nz_sim.reshape(-1,1)
    orientation=np.hstack([nx_sim,ny_sim,nz_sim])
    Para4 = leastsq(residuals2, x0=[-0.9, 0, 0, 0, 20], args=(orientation[0:50], T_sim[0:50]), maxfev=10000)
    print(Para4)
    #Para4,_ = curve_fit(func2, T_sim,orientation )
    plt.plot(T_sim, func2(T_sim, Para4[0])[:, 0], linewidth=2)
    plt.plot(T_sim, func2(T_sim, Para4[0])[:, 1], linewidth=2)
    plt.plot(T_sim, func2(T_sim, Para4[0])[:, 2], linewidth=2)

    plt.show()

def calculate_catch_point(para, method):
    methods = {
        'plane': calculate_catch_point_plane
    }
    if methods in methods:
        return methods[method](para)

def calculate_catch_point_plane(para):
    return solve_time_period2(theta=para[0], robot_loc=para[1], robot_range=para[2], zcatch=para[3])

if __name__ == '__main__':
    main()
    #test()
