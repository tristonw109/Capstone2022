from cgitb import scanvars
import os
from math import cos, sin, pi, floor, sqrt
from turtle import distance
import pygame
from adafruit_rplidar import RPLidar
import time
import numpy as np
import struct
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import threading
import RPi.GPIO as GPIO
import argparse
import socket
import random
import sys
global epsValue 


# Console Arguments
"""
    Two required arguments to the program
    -p : Port Number for server. Must be the same as client
    -d : Direction. Robot can go left or right in the main scenario.
"""
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="specify port number",
                     required="true")
parser.add_argument("-d", "--direction", help="specify direction",
                     required="true")

args = parser.parse_args()
direction = args.direction

# ==========================================================================================

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=25)

# ==========================================================================================

# Set up Network Server

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print ("Socket successfully created")
except socket.error as err:
    print ("socket creation failed with error %s" %(err))
        
print ("ROBOT A successfully created")

hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
print(ip)
portNum = int(args.port)
direction = args.direction
s.bind(('192.168.4.1', portNum))        
print ("ROBOT A binded to %s" %(portNum))
  
 # put the socket into listening mode
s.listen(5)    
print ("ROBOT A is listening")           
 
c, addr = s.accept()    



# ==========================================================================================


mainSem = threading.Semaphore(1)
stoppingSem = threading.Semaphore(1)

stoppingSem.acquire()



# Setting up GPIOS being used
# =================================================
GPIO.setmode(GPIO.BCM)

#set up GPIO pins
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24,GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21,GPIO.OUT)

#DL Communication
GPIO.setup(26,GPIO.IN)

#GPIO PINS 5 & 0 are Linetrackers
GPIO.setup(5,GPIO.IN)
GPIO.setup(0,GPIO.IN)

pwma = GPIO.PWM(23,20)
pwmb = GPIO.PWM(20,20)

#RGB Led
#Red
GPIO.setup(11,GPIO.OUT)
#Green
GPIO.setup(9,GPIO.OUT)
#Blue
GPIO.setup(10,GPIO.OUT)


#==================================================

# Declaration of variables

scan_data = [0]*360
timerStart = time.time()
filteredAngles = np.zeros([1,1])
filteredDist = np.zeros([1,1])

lastStopRecieved = 0

timerStart = time.time()

signSawTime = 0

motorSpeed = 10

distanceSeenArray = np.zeros([1,1])

timeArray = np.zeros([1,1])

actualDistanceSeenArray = np.zeros([1,1])

speedArray = np.zeros([1,1])

timeUntilArray = np.zeros([1,1])

epsilonArray = np.zeros([1,1])

speed = 0.0000000000001

timeUntilIntersection = 999999

avgSpeed = 0.000000000001

widthChecker = 0

flipper = 0


# Global 

global lastSignDistance
lastSignDistance = 0

global widthArray
widthArray = np.zeros([1,1])

global distanceArray
distanceArray = np.zeros([1,1])

global baton
baton = 0

global scanCount
scanCount = 0

global batonPassOffTime
batonPassOffTime = time.time()

global myBatonTime
myBatonTime = time.time()

global obstructionTime
obstructionTime = time.time()

global lineDisabledTime
lineDisabledTime = time.time()

global timeStamp
timeStamp = time.time()

global lastTimeUntilIntersection
lastTimeUntilIntersection = 9999

# =======================================================

"""
    Motor Functions. These were inherited.
    Adjusting the motor code is necessary if the motor driver or motors change

    Here is a list of the functions:
        setupPorts()
        cleanPorts()
        motorOn()
        motorOnF()
        motorOnSlow()
        slow()
        forward()
        backwards()
        rightturn()
        leftturn()
        stop()

"""

def setupPorts():
        #Declares the GPIO settings
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)



        #set up GPIO pins
        GPIO.setup(23, GPIO.OUT)
        GPIO.setup(24,GPIO.OUT)
        GPIO.setup(20, GPIO.OUT)
        GPIO.setup(21,GPIO.OUT)
        
def cleanPorts():
        # Reset all the GPIO pins by setting them to LOW
        GPIO.output(23, GPIO.LOW) # Set AIN1
        GPIO.output(24, GPIO.LOW) # Set AIN2
        GPIO.output(20, GPIO.LOW) # Set AIN1
        GPIO.output(21, GPIO.LOW) # Set AIN2
                
        
def motorOn():
        #time.sleep(0.01)
        startTime = time.time()
        pwma.start(3)
        pwmb.start(3)        
        while((time.time()-startTime) < 0.025):
            pass

def motorOnF():
        #time.sleep(0.01)
        startTime = time.time()
        GPIO.output(20, GPIO.HIGH)
        pwma.start(3)
        #GPIO.output(20, GPIO.HIGH)
        while((time.time()-startTime) < 0.035):
            pass

def motorOnSlow():
        #time.sleep(0.01)
        startTime = time.time()
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(20, GPIO.HIGH)
        while((time.time()-startTime) < 0.020):
            pass


def slow():
        setupPorts()
        GPIO.setwarnings(False)

        #turns the motors on


        GPIO.output(23,GPIO.LOW)
        GPIO.output(24,GPIO.HIGH)
        
        GPIO.output(20,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)   


        motorOnSlow()
    
        cleanPorts()
        cleanPorts()
        
def forward():
        setupPorts()
        GPIO.setwarnings(False)

        #turns the motors on


        GPIO.output(23,GPIO.LOW)
        GPIO.output(24,GPIO.HIGH)
        
        GPIO.output(20,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)   


        motorOn()
    
        cleanPorts()
        cleanPorts()

def rightturn():
        
        setupPorts()
        GPIO.setwarnings(False)
        #print("Going Right")
        GPIO.output(23,GPIO.HIGH)
        GPIO.output(24,GPIO.LOW)
        
        GPIO.output(20,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)     
        
        motorOn()

        
        cleanPorts()  

def backwards():
    setupPorts()
    #print("Going Backwards")
       
    GPIO.output(23,GPIO.LOW)
    GPIO.output(24,GPIO.HIGH)
    GPIO.output(20,GPIO.LOW)
    GPIO.output(21,GPIO.HIGH)
                    
    motorOn()

    cleanPorts()
        
def leftturn():
    
        #print("Left Turn")
        GPIO.setwarnings(False)

        setupPorts()


        GPIO.output(23,GPIO.LOW)
        GPIO.output(24,GPIO.HIGH)
        
        GPIO.output(20,GPIO.LOW)
        GPIO.output(21,GPIO.HIGH) 
        
        motorOn()

        cleanPorts()
        
def stop():
    setupPorts()
    GPIO.setwarnings(False)
    #print("Stopping")

    setupPorts()
    
    GPIO.output(23,GPIO.LOW)
    GPIO.output(24,GPIO.LOW)
    GPIO.output(20,GPIO.LOW)
    GPIO.output(21,GPIO.LOW)    
    pwma.start(0)
    pwmb.start(0)  
    cleanPorts()


# ============================================================================================





# This processes the scan data right infront of the robot
# This is used for an emergency stop
def middleQuadrantScan(scan):
    global obstructionTime
    for (_, angle, distance) in scan:
        scan_data[min([359, floor(angle)])] = distance
        # Filtering down to certain angles and distances
        if ((distance > 50) and (distance < 300)) and ((angle >65) and (angle < 125)):
            obstructionTime = time.time()
            return 1
    return 0




"""
    This thread, thread1, performs the DBSCANs ALG.
    It flips between 2 Epsilon values, 55 and 30.
    These may be adjusted depending on the scenario in which the code is deployed.

"""
def thread1(scan,):

    scan_data = [0]*360

    for (_, angle, distance) in scan:
        scan_data[min([359, floor(angle)])] = distance


    global timerStart
    global flipper
    global epsValue

    if((time.time() - timerStart) > 0.050):

        intakeTime = time.time()

        clusterDataset = scan[0:360]
        global scanCount
        scanCount = scanCount + 1
        
        filteredAngles = np.zeros([1,1])
        filteredDist = np.zeros([1,1])
        clusterAngle = (list(zip(*clusterDataset))[1])
        clusterDist = (list(zip(*clusterDataset))[2])
        for k in range(len(clusterAngle)):
            if(clusterAngle[k] < 165 and clusterAngle[k] > 90):
                if(clusterDist[k] > 1 and clusterDist[k] < 2500):
                    filteredAngles = np.insert(filteredAngles,0, clusterAngle[k],axis=0)
                    filteredDist = np.insert(filteredDist,0,[clusterDist[k]],axis=0)

        timerStart = time.time()
        
        #changing angles to radians
        filteredAngles = np.multiply(filteredAngles,(np.pi/180))

        #converting to cartesian
        c = abs(np.cos(filteredAngles))
        s = abs(np.sin(filteredAngles))

        X = np.multiply(filteredDist,c)
        Y = np.multiply(filteredDist,s)

        #creating 2D numpy array for processing
        P = np.column_stack((X,Y))

        epsValue = 55
        # DBscan Algorithm
        if ((flipper % 2) == 1):
            epsValue = 30
            
        # Flip between the two epsilon values of 30 & 55
        
        flipper += 1
        db = DBSCAN(eps=epsValue, min_samples=3).fit(P)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        global widthArray
        global distanceArray
        midpointArray = np.zeros([2,2])
        widthArray = np.zeros([1,1])
        distanceArray = np.zeros([1,1])
        
        finalMid = np.zeros([2,2])
        finalWidth = np.zeros([1,1])
        finalDist = np.zeros([1,1])



        xy = np.zeros([2,2])
        PP = np.zeros([2,2])
        unique_labels = set(labels)
        


        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for j, col in zip(unique_labels,colors):

            if (j != -1):

                
                class_member_mask = labels == j
                if ((np.shape((P[class_member_mask & ~core_samples_mask])))[0] != 0):
                    xy = np.concatenate((xy,((P[class_member_mask & ~core_samples_mask])[[0,-1]])),axis=0)
                else:
                    n_clusters_ = n_clusters_-1


        xy = np.delete(xy,0, axis=0)
        xy = np.delete(xy,0, axis=0)


        for k in range(n_clusters_):
            iterator2 = 2*k
            iterator1 = 2*k+1
            x_val = (((xy[(iterator1),0]+xy[(iterator2),0])/2))
            y_val = (((xy[(iterator1),1]+xy[(iterator2),1])/2))
            midpointArray = np.insert(midpointArray,0, [x_val,y_val],axis=0)
            width_val = sqrt(((xy[(iterator2),0] - xy[(iterator1),0])**2)+((xy[(iterator2),1] - xy[(iterator1),1])**2))
            widthArray = np.insert(widthArray,0,width_val,axis=0)
            dist_val = sqrt(((x_val-0)**2)+(((y_val-0))**2))
            distanceArray = np.insert(distanceArray,0,dist_val,axis=0)

        midpointArray = np.delete(midpointArray,(midpointArray.shape[0])-1,axis=0)
        midpointArray = np.delete(midpointArray,(midpointArray.shape[0])-1,axis=0)
        widthArray = np.delete(widthArray,(widthArray.shape[0])-1,axis=0)
        distanceArray = np.delete(distanceArray,(distanceArray.shape[0])-1,axis=0)


        """
            Uncomment these if you want more information outputted from the clustering ALG
        """
        #print("MIDPOINTS:")
        #print(midpointArray)
        #print("WIDTH:")
        #print(widthArray)
        #print("DISTANCE FROM ORIGIN:")
        #print(distanceArray)

        #print("Time Taken: ")
        #print(time.time() - intakeTime)


"""
    Thread 2 concerns the network communication between the two robots
    This robot acts as the server.
    
"""
def thread2(c, timeUntilIntersection):

    #Ignore this the first time
    global lastTimeUntilIntersection
    if timeUntilIntersection < 5 and timeUntilIntersection != lastTimeUntilIntersection:
        #Take snapshot of time. Should only happen once. 
        global timeStamp
        timeStamp = time.time()

    #Reassign this value with the new timeuntilIntersection.
    lastTimeUntilIntersection = timeUntilIntersection

    if (time.time()-timeStamp) > timeUntilIntersection:
        robotPassed = True
    else:
        robotPassed = False
    


    global baton
    if(baton == 1 or baton == 2):
        #If one has the baton, send useless information
        Robot_A_Packet = struct.pack('fib',999.999, baton, robotPassed)
    else:
        #If neither has the baton communicate relevant information
        Robot_A_Packet = struct.pack('fib',timeUntilIntersection, baton, robotPassed)
    c.setblocking(1)
    c.send(Robot_A_Packet)
    
    Robot_B_Packet = (c.recv(struct.calcsize('fib')))
    Robot_B_Info = struct.unpack('fib', Robot_B_Packet)
    time_until_int_b = Robot_B_Info[0]
    robotPassedRecieved = Robot_B_Info[2]

    if ((baton != 2) and (Robot_B_Info[1] == 2)):
        global batonPassOffTime
        batonPassOffTime = time.time()




        
    lastBaton = baton
    baton = Robot_B_Info[1]


    if((time_until_int_b > timeUntilIntersection) and (time_until_int_b < 5 and timeUntilIntersection < 5) and robotPassed == False and robotPassedRecieved == False and (baton == 0)):
        #Robot A gets the Baton
        global myBatonTime
        myBatonTime = time.time()
        baton = 1
    elif((time_until_int_b == timeUntilIntersection) and (time_until_int_b < 5 and timeUntilIntersection < 5) and robotPassed == False and robotPassedRecieved == False and (baton == 0)):
        #Robot A gets it cuz i said so
        baton = 1
    else:
        #Doesn't have the baton
        if(lastBaton != baton and baton != 0 and lastBaton != 0 ):
            baton = 1
        else:
            baton = baton



#Handles the RGB indicator
def thread3():
    global baton
    global batonPassOffTime
    global myBatonTime

    if baton == 0:
        #RGB
        #GREEN
        GPIO.output(11,GPIO.HIGH)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.LOW)   
	
    elif baton == 1 and (time.time()- myBatonTime) < 10:
        #RGB
        #GREEN
        GPIO.output(11,GPIO.LOW)
        GPIO.output(9,GPIO.HIGH)
        GPIO.output(10,GPIO.LOW)        
    elif baton == 2 and (time.time()- batonPassOffTime) < 3:
        #RGB
        #BLUE
        GPIO.output(11,GPIO.LOW)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.HIGH)
    else:
        #RGB
        GPIO.output(11,GPIO.HIGH)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.LOW)


# ===================================================================================
"""
    Main Thread
    Handles:
    activating the clustering w/ use of communication to DL Pi through GPIO
    Calculates Speed
    Motor Control and Emergency Stop
    Control of the other threads

"""
try:

    for scan in lidar.iter_scans():
        #Motor Code
        
        
        #print(lastStopRecieved)
        
        widthChecker = 0
        dlStop = GPIO.input(26)
        for num in range(len(widthArray)):
            dist = distanceArray[num]
            adjustedWidthThreshold = 2.082*pow(10,-8)*pow(dist,3)-5.32*pow(10,-5)*pow(dist,2)-0.03107*dist+182.8
            #print(adjustedWidthThreshold)
            if((widthArray[num] >= adjustedWidthThreshold and widthArray[num] <= 210)):
                widthChecker = 1
                signSawTime = time.time()
                lastSignDistance = distanceArray[num]
                if distanceArray[num] > 0.85*distanceSeenArray[len(distanceSeenArray) - 1] and distanceArray[num] < 1.02*distanceSeenArray[len(distanceSeenArray) - 1] or len(distanceSeenArray) == 1:
                    distanceSeenArray = np.append(distanceSeenArray, lastSignDistance)
                    timeArray = np.append(timeArray, signSawTime)
                    print(distanceSeenArray)
                    print(timeArray)

                    # This is calculating the moving average of SPEED
                    # It starts after recieving 4 data points
                    for i in range(len(distanceSeenArray)-4):
                        if(distanceSeenArray[i] != 0):
                            speed1 = abs((distanceSeenArray[i]-distanceSeenArray[i+1])/(timeArray[i]-timeArray[i+1]))
                            speed2 = abs((distanceSeenArray[i+1]-distanceSeenArray[i+2])/(timeArray[i+1]-timeArray[i+2]))
                            speed3 = abs((distanceSeenArray[i]-distanceSeenArray[i+4])/(timeArray[i]-timeArray[i+4]))
                            speed4 = abs((distanceSeenArray[i+2]-distanceSeenArray[i+3])/(timeArray[i+2]-timeArray[i+3]))
                            speed5 = abs((distanceSeenArray[i+3]-distanceSeenArray[i+4])/(timeArray[i+3]-timeArray[i+4]))
                            avgSpeed = (speed1+speed2+speed3+speed4+speed5)/5
                        
                    actualSignDistance = sqrt(abs(pow(lastSignDistance,2)-pow(558.8,2)))

                    if(avgSpeed < 100):
                        timeUntilIntersection = (actualSignDistance/avgSpeed)           
                    else:
                    
                        timeUntilIntersection = (actualSignDistance/avgSpeed)
                    print(avgSpeed)
                    print("    ")
                    print("timeUntilIntersection:")
                    print(timeUntilIntersection) 
                    actualDistanceSeenArray = np.append(actualDistanceSeenArray, actualSignDistance)
                    speedArray = np.append(speedArray,avgSpeed)
                    timeUntilArray = np.append(timeUntilArray, timeUntilIntersection)
                    epsilonArray = np.append(epsilonArray,epsValue)
            
               
        leftLineTracker = GPIO.input(5)
        rightLineTracker = GPIO.input(0)
        
        # If direction is left, and both line trackers go high, ignore the right line tracker for 2 seconds
        if(direction == 'left'):
            if((leftLineTracker == 1 and rightLineTracker == 1) and (time.time() - lineDisabledTime) > 3):
                lineDisabledTime = time.time()
            if(middleQuadrantScan(scan) == 1) or ((time.time()-obstructionTime) < 2.5):
                stop()
            elif(leftLineTracker == 1):
                rightturn()
            elif(rightLineTracker == 1 and ((time.time() - lineDisabledTime) > 2)):
                leftturn()

            elif(baton == 2 and ((time.time()- batonPassOffTime) < 3)):
                slow()
                
            else:
                forward()

        # If direction is not left, and both line trackers go high, ignore the left line tracker for 2 seconds
        else:
            if((leftLineTracker == 1 and rightLineTracker == 1) and (time.time() - lineDisabledTime) > 3):
                lineDisabledTime = time.time()
            if(middleQuadrantScan(scan) == 1) or ((time.time()-obstructionTime) < 2.5):
                stop()
            elif(rightLineTracker == 1):
                leftturn()
            elif(leftLineTracker == 1 and ((time.time() - lineDisabledTime) > 2)):
                rightturn()
            elif(baton == 2 and ((time.time()- batonPassOffTime) < 3)):
                slow()
                
            else:
                forward()
        
        #Starting of all threads and sending their appropriate parameters
        b = threading.Thread(target = thread1, args = (scan,))
        a = threading.Thread(target = thread2, args = (c,timeUntilIntersection,))
        d = threading.Thread(target = thread3, args = ())
        b.start()
        a.start()
        d.start()
            

# In the case of control+c, perform these actions for a 'cleaner' shutdown
except KeyboardInterrupt:
    print("Stopping.")
    cleanPorts()
    pwma.stop()
    pwmb.stop()
    lidar.stop()
    lidar.disconnect()
