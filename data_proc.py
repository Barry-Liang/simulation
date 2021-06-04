import numpy as np
import matplotlib.pyplot as plt
from Pre_n_move import get_pre_param2, solve_time_period2, solve_time_period3
from ring_pre_catch_mini import *
from scipy.fftpack import fft, fftfreq


def plot_dis(data, time):
    position_data = data[:, :3]
    position_data[:, 2] = map(lambda x: x-1, position_data[:,2])
    data_num = len(time)
    robot_location = (0.505, 0.0, 0.0)
    robot_reach = 1.0
    catch_positions = []
    for i in range(6, data_num):
        theta = get_pre_param2(position_data[:i, :], time[:i])
        t1 = solve_time_period2(theta=theta, robot_loc=robot_location, robot_range=robot_reach, zcatch=0.60)
        t2 = solve_time_period3(theta=theta, robot_loc=robot_location, robot_range=robot_reach, catch_ratio=0.9)
        if t1 == -1 or t2 == -1 or (t1 < t2):
            # catch_position.append((0, 0, 0))
            continue

        tcatch, distance = catch_point_least_cartesian_distance(t1, t2, theta)
        catch_position = time_to_loc(theta, tcatch)
        catch_positions.append(catch_position)

    print(len(catch_positions))
    last_catch_position = catch_positions[-1]
    distance = []
    for catch_position in catch_positions:
        distance.append(math.sqrt((catch_position[0] - last_catch_position[0]) ** 2
                                  + (catch_position[1] - last_catch_position[1]) ** 2
                                  + (catch_position[2] - last_catch_position[2]) ** 2))

    plt.plot(distance)
    plt.show()


path = '/home/hairui/Downloads/data/data/'
PoseSet = np.load(path + 'camera_result-perfect.npy')
TimeSet = np.load(path + 'time_result-perfect.npy')
plot_number = len(TimeSet)
print(plot_number)
index = np.where(PoseSet[:plot_number, 5] > 0)
PoseSet[index, 3] = -PoseSet[index, 3]
PoseSet[index, 4] = -PoseSet[index, 4]
PoseSet[index, 5] = -PoseSet[index, 5]
norm = np.sqrt(PoseSet[:plot_number, 3] ** 2 + PoseSet[:plot_number, 4] ** 2 + PoseSet[:plot_number, 5] ** 2)
nx = PoseSet[:plot_number, 3] / norm
ny = PoseSet[:plot_number, 4] / norm
nz = PoseSet[:plot_number, 5] / norm


nxf = fft(nx)
nxf_real = nxf.real
nxf_imag = nxf.imag

nxf = abs(nxf)
print(len(nxf))
nxf1 = nxf/len(nx)
nxf2 = nxf1[range(int(len(nx)/2))]

xf = fftfreq(len(nx), TimeSet[-1]/len(nx))
xf1 = xf
xf2 = xf[range(int(len(nx)/2))]

plt.plot(xf2, nxf2)
plt.show()
