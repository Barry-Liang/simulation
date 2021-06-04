import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


pose = np.load('/home/liangxiao/Documents/data/camera_result-perfect.npy')
time = np.load('/home/liangxiao/Documents//data/time_result-perfect.npy')

print (pose.shape)
len_p = pose.shape[0]

def time_to_loc(theta, time):
    x = theta[0] + theta[1] * time
    y = theta[2] + theta[3] * time
    z = theta[4] + theta[5] * time + theta[6] * time * time
    position = (x, y, z)
    return position

def get_pre_param2(position_set, time_series): #input position array, return x = a0 + a1*t y = a2 + a3*t z = a4 + a5*t + a6*t*t
    time = time_series.reshape(-1, 1)
    positionx = position_set[:, 0].reshape(-1, 1)
    positiony = position_set[:, 1].reshape(-1, 1)
    positionz = position_set[:, 2].reshape(-1, 1)
    modelx = LinearRegression(fit_intercept=True)
    modely = LinearRegression(fit_intercept=True)
    modelz = LinearRegression(fit_intercept=True)

    second_time = np.power(time, 2) * (-4.9)
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

theta = get_pre_param2(pose[:, :3], time)
print (theta)

nihe = time_to_loc(theta, time)

plt.figure(1, figsize=(9,6))
plt.scatter(time[:], pose[:, 0], color='darkred', label='x',marker = '^')
plt.plot(time, nihe[0], label='Trajectory in x',color='yellow')
plt.scatter(time[:], pose[:, 1], label='y', marker='o',color='green')
plt.plot(time, nihe[1], color='pink', label = 'Trajectory in y')
plt.scatter(time[:], pose[:, 2], label='z', marker='p')
plt.plot(time, nihe[2], color='orange', label = 'Trajectory in z')
plt.xlabel('Time (s)')
plt.ylabel('Location (m)')
plt.legend()





nihe = np.array(nihe).transpose()
print (np.sqrt(np.sum(np.power(nihe-pose[:,:3], 2), axis=0)/nihe.shape[0]))
k = 20
error_w_time = np.zeros((len_p-k, 3))
error_all = np.zeros((len_p-k))
for i in range(k, len_p):
    theta = get_pre_param2(pose[:i,:3],time[:i])
    nihe = time_to_loc(theta, time)
    nihe = np.array(nihe).transpose()
    print(pose.shape, time.shape, nihe.shape)
    error_w_time[i-k,:] = np.sqrt(np.sum(np.power(nihe - pose[:, :3], 2), axis=0) / nihe.shape[0])
error_all = np.sqrt(np.sum(np.power(error_w_time[:,:], 2), axis=1))
plt.figure(2)
plt.plot(time[k:], error_w_time[:,0], label='x')
plt.plot(time[k:], error_w_time[:,1], label='y')
plt.plot(time[k:], error_w_time[:,2], label='z')
plt.plot(time[k:], error_all[:], label='Overall')
plt.xlabel('Time (s)')
plt.ylabel('RMSE of Predicted Trajectory (m)')
plt.legend()




index = np.where(pose[:,5]<0)
pose[index,5] = - pose[index, 5]
pose[index,4] = - pose[index, 4]
pose[index,3] = - pose[index, 3]
pose_norm = np.sqrt(np.power(pose[:,3],2)+np.power(pose[:,4],2)+np.power(pose[:,5],2))
pose[:,3] = pose[:,3]/pose_norm
pose[:,4] = pose[:,4]/pose_norm
pose[:,5] = pose[:,5]/pose_norm
print (np.max(pose[:,5]))
plt.figure(3)
i = 5
num = 0
plt.scatter(time[:], pose[:, 3], label='nx')
plt.scatter(time[:], pose[:, 4], label='ny')
plt.scatter(time[:], pose[:, 5], label='nz')
time2 = np.vstack([time, time, time]).T

from scipy.optimize import leastsq

# def func(x,p):
#     A,k,theta, b = p
#     return A*np.sin(2*np.pi*k*x+theta)+b
# def residuals(p,y,x):
#     return y-func(x, p)
# max = np.max(pose[:,i])
# Para1 = leastsq(residuals, x0=[0.01, 4.8 ,0.1, 0.0], args=(pose[10:, 3], time[10:]))
# Para2 = leastsq(residuals, x0=[0.1, 4.8 ,10, 0.5], args=(pose[10:, 4], time[10:]))
# Para3 = leastsq(residuals, x0=[0.01, 4.8 ,0.1, 0.0], args=(pose[10:, 5], time[10:]))


plt.xlabel('Time (s)')
plt.ylabel('Or/home/chenyl/ws_mainientation (1)')



def func2(x,p):
    A,a1,a2,a3,a4 = p
    theta = -a4*x + a3
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
print(pose.shape)
print(time.shape)
Para4 = leastsq(residuals2, x0=[0.9, 0, 0, 0, 2*np.pi*4.9], args=(pose[20:, 3:6], time[20:]),maxfev=100000)

#print(np.sum(np.power(func2(time[:], Para4[0])[5,:3], 2)))
print (Para4[0])

plt.plot(time[:], func2(time[:], Para4[0])[:,0], linewidth=2)
plt.plot(time[:], func2(time[:], Para4[0])[:,1], linewidth=2)
plt.plot(time[:], func2(time[:], Para4[0])[:,2], linewidth=2)
plt.show()


