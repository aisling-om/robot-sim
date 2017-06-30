#! /usr/bin/env python

"""
Simple extended (nonlinear) Kalman filter for test data (accelerometer & gyroscope).

Assume simple measurement and state transition errors without cross-terms

Large state error (relative to measurement error) will tend to follow measurement
Small state error will tend to be smooth and insensitive to measurement
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

fact = np.pi/180. #degrees to radians

#these two constants define the performance of the filter 
state_err = 1.E-1
meas_err = 5.E-3

# EKF function
def EKFalg (t,z):
	xhat = np.zeros((2, len(t)))
	
	#state is [phi, phidot]
  	state = np.array([[0],[0]])
  	P = 1.E5*np.array([[1,0],[0,1]]) #initialize covariance
  	previous_t = 0;
  
  	#Error matrices, assume constant and no cross-terms
	R = meas_err*np.eye(3) #measurement error
	Q = state_err*np.eye(2) #state transition error
	
	for i in range(len(t)):
		dt = t[i] - previous_t
		A = np.array([[1,dt],[0,1]]) #state transition matrix

		#Observation
		z_t = np.array([z[i,:]]).T

		
		xprob = A.dot(state)
		pprob = A.dot(P.dot(A.T)) + Q
		
		#Measurement function h & Jacobian of measurement function jac_h
		h = np.array([[np.sin(xprob[0,0]*fact)], [np.cos(xprob[0,0]*fact)], [xprob[1,0]]])
		jac_h = np.array([ [fact*np.cos(state[0,0]*fact),0], [-1*fact*np.sin(state[0,0]*fact),0], [0,1] ])	
		tjac_h = jac_h.T

		#Kalman gain		
		k1 = np.linalg.inv(jac_h.dot(pprob.dot(tjac_h)) +R) #inner term		
		K = pprob.dot(tjac_h.dot(k1))

		#update state
		xhat_i = xprob + K.dot(z_t-h)

		#update covariance
		P1 = np.eye(2) - K.dot(jac_h)
		P = P1.dot(pprob)

		state = xhat_i
		xhat[:,i] = state[:,0]
		
		previous_t = t[i]

	return xhat

#Load and prepare data
mat = scipy.io.loadmat('data.mat') #file is Matlab v5.0 so works
#print mat
trial = mat['trial0']
time = trial[:,0]/1000.
ay = trial[:,1]/16384. #y-accelerometer - sin(phi)
az = trial[:,2]/16384. #z-accelerometer - cos(phi)
gx = trial[:,3]/131. #gyroscope - dphi/dt

ze = np.vstack([ay,az,gx])
ze = ze.T #z[i] is (ay[i], az[i], gx[i])

nel = len(time);
  
time = time[0:nel]
ze = ze[0:nel,:]
gx = gx[0:nel]
ay = ay[0:nel]
az = az[0:nel]

#Call EKF on time and sensor measurement  
xhat = EKFalg(time, ze);

#Predicted acceleration in y- and z-directions
accelPred = np.zeros((2,len(time)))
for i in range(len(time)):
	accelPred[0,i] = np.sin(xhat[0,i]*fact)
	accelPred[1,i] = np.cos(xhat[0,i]*fact)

#Plot
ax = plt.subplot(4, 1, 1)
plt.plot(time,ay, 'r-', label="Data")
plt.plot(time, accelPred[0,:], 'b-', label = "Filter")
plt.ylabel("$a_{y}$")

ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.4), fancybox=True, shadow=True)

plt.subplot(4,1,2)
plt.plot(time, az, 'r-',time, accelPred[1,:], 'b-')
plt.ylabel("$a_{z}$")

plt.subplot(4,1,3)
plt.plot(time, xhat[1,:], 'b-',time, gx, 'r--')
plt.ylabel("$\dot\phi$")

plt.subplot(4,1,4)
plt.plot(time, xhat[0,:], 'b-')
plt.ylabel("$\phi$")

plt.show()