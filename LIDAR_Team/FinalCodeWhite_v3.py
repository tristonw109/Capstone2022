import os
from math import cos, sin, pi, floor, sqrt
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

# ===========================================
# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME)
# =====================================================================
"""
    Set up console parameter
    - p : Port for client to connect to server on.
    No direction in this one as this robot is designed to only go forward.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="specify port number",required="true")

args = parser.parse_args()

# =================================================================================

mainSem = threading.Semaphore(1)
stoppingSem = threading.Semaphore(1)

stoppingSem.acquire()

#Take in console arguments

# =====================================================
"""
    set up GPIO pins
"""
GPIO.setmode(GPIO.BCM)

GPIO.setup(23, GPIO.OUT)
GPIO.setup(24,GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21,GPIO.OUT)

pwma = GPIO.PWM(23,20)
pwmb = GPIO.PWM(20,20)

#RGB Led
#Red
GPIO.setup(11,GPIO.OUT)
#Green
GPIO.setup(9,GPIO.OUT)
#Blue
GPIO.setup(10,GPIO.OUT)

#GPIO PINS 5 left & 0 right are Linetrackers

GPIO.setup(5, GPIO.IN)
GPIO.setup(0, GPIO.IN)

#DL communication
GPIO.setup(26, GPIO.IN)



# ==========================================

# Variable Declaration



scan_data = [0]*360


timerStart = time.time()
filteredAngles = np.zeros([1,1])
filteredDist = np.zeros([1,1])

lastStopRecieved = 0

widthChecker = 0

#Speed Variable
motorSpeed = 9

timerStart = time.time()
signSawTime = 0

distanceSeenArray = np.zeros([1,1])
timeArray = np.zeros([1,1])
speed = 0.0000000001
timeUntilIntersection = 999999999
avgSpeed = 0.00000000001


#Global Declarations

global batonPassOffTime
batonPassOffTime = time.time()

global myBatonTime
myBatonTime = time.time()

global obstructionTime
obstructionTime = time.time()

global timeStamp
timeStamp = time.time()

global lastTimeUntilIntersection
lastTimeUntilIntersection = 9999

global robotPassed
robotPassed = False

global estimatedTime
estimatedTime = 999999999999

global lastSignDistance
lastSignDistance = 0

global widthArray
widthArray = np.zeros([1,1])

global distanceArray
distanceArray = np.zeros([1,1])

global baton
baton = 0

# ============================================================

#This processes the scan data right infront of the robot
def middleQuadrantScan(scan):
    global obstructionTime
    for (_, angle, distance) in scan:
        scan_data[min([359, floor(angle)])] = distance
        # Filtered down for the disance and angles we care about
        if ((distance > 50) and (distance < 500)) and ((angle >80) and (angle < 110)):
            obstructionTime = time.time()
            return 1
    return 0

# ===========================================================================
"""
    Motor Code Section
    Most of these were inherited from the previous capstone group w/ some tweaks.
    If the motors or motor drivers are changed -  the code must change.
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
        pwma.start(15)#was 15
        pwmb.start(15)#waas 15        
        #while((time.time()-startTime) < 0.025):
        while((time.time()-startTime) < 0.1):
            pass
def motorOnF(motorSpeed):
        #time.sleep(0.01)
        startTime = time.time()
        pwma.start(motorSpeed)
        pwmb.start(motorSpeed)        
        while((time.time()-startTime) < 0.0010):
            pass
#         print("Motor On")
        
        
def forward(motorSpeed):
        setupPorts()
        GPIO.setwarnings(False)

        #turns the motors on

        GPIO.output(23,GPIO.HIGH)
        GPIO.output(24,GPIO.LOW)
        
        GPIO.output(20,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)




        motorOnF(motorSpeed)
    
        cleanPorts()

def rightturn():
        
        setupPorts()
        GPIO.setwarnings(False)
        #print("Going Right")
        GPIO.output(23,GPIO.LOW)
        GPIO.output(24,GPIO.HIGH)
        
        GPIO.output(20,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)        
        
        motorOn()

        
        cleanPorts()        
def backwards():
    setupPorts()
    print("Going Backwards")
       
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
        GPIO.output(23,GPIO.HIGH)
        GPIO.output(24,GPIO.LOW)
        
        GPIO.output(20,GPIO.LOW)
        GPIO.output(21,GPIO.HIGH)        

        
        motorOn()

        cleanPorts()
        
def stop():
    setupPorts()
    GPIO.setwarnings(False)
    print("Stopping")

    setupPorts()
    
    GPIO.output(23,GPIO.LOW)
    GPIO.output(24,GPIO.LOW)
    GPIO.output(20,GPIO.LOW)
    GPIO.output(21,GPIO.LOW)    
    motorOn()
    cleanPorts()

# ==================================================================================

def thread1(scan,):

    scan_data = [0]*360

    for (_, angle, distance) in scan:
        scan_data[min([359, floor(angle)])] = distance


    global timerStart
    global flipper


    # How often we run the alg. Depends on the hardware how often you can run this
    if((time.time() - timerStart) > 0.1):

        intakeTime = time.time()

        clusterDataset = scan[0:360]
        filteredAngles = np.zeros([1,1])
        filteredDist = np.zeros([1,1])
        clusterAngle = (list(zip(*clusterDataset))[1])
        clusterDist = (list(zip(*clusterDataset))[2])
        for k in range(len(clusterAngle)):
            # Filtering down to the angles and distances we care about, in this case 1st quadrant.
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
        if ((flipper % 3) == 1):
            epsValue = 30
        elif((flipper % 3) == 0):
            epsValue = 55
            
        
        #print("___________________________")
        #print("Eps Value: ",epsValue)
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
            Uncomment these to display the important outputs of clustering.
        """
        #print("MIDPOINTS:")
        #print(midpointArray)
        #print("WIDTH:")
        #print(widthArray)
        #print("DISTANCE FROM ORIGIN:")
        #print(distanceArray)

        #print("Time Taken: ")
        #print(time.time() - intakeTime)

# ==========================================================
"""
    Thread2 is the networking thread, starts after we initially connect to the server in the main thread, this starts
    The white robot is the client robot, that is why this thread looks *slightly* different from the gold's version of
    this thread.
"""
def thread2(s,timeUntilIntersection):
    Robot_A_Packet = (s.recv(struct.calcsize('fib')))
    global timeStamp
    global lastTimeUntilIntersection
    global estimatedTime 
    Robot_A_Info = struct.unpack('fib',Robot_A_Packet)
    robotPassedRecieved = Robot_A_Info[2]
    if timeUntilIntersection < 5 and timeUntilIntersection != lastTimeUntilIntersection:
        #Take snapshot of time. Should only happen once. 
        
        timeStamp = time.time()
        print("New robot Passed Time")
        print(timeUntilIntersection)
        estimatedTime = timeUntilIntersection

    #Reassign this value with the new timeuntilIntersection.
    lastTimeUntilIntersection = timeUntilIntersection

    global robotPassed
    
    if (time.time()-timeStamp) > estimatedTime:
        robotPassed = True
    elif (robotPassed == True):
        robotPassed = True
    else:
        robotPassed = False

    print(robotPassed)


    global baton
    if(baton == 1 or baton == 2):

        Robot_B_Packet = struct.pack('fib',999.99,baton, robotPassed)

    else:
         Robot_B_Packet = struct.pack('fib',timeUntilIntersection,baton, robotPassed)

    s.send(Robot_B_Packet)
    time_until_int_a = Robot_A_Info[0]

    if ((baton != 1) and (Robot_A_Info[1] == 1)):
        global batonPassOffTime
        batonPassOffTime = time.time()
    baton = Robot_A_Info[1]

    if ((time_until_int_a > timeUntilIntersection) and (time_until_int_a < 5 and timeUntilIntersection < 5) and robotPassed == False and robotPassedRecieved == False and (baton == 0)):

        global myBatonTime
        myBatonTime = time.time()
        baton = 2
    else:
        #Doesn't have the baton
        baton = baton

    print(baton)
    
    
# This thread handles the RGB on the back
def thread3():
    global baton
    global batonPassOffTime
    global myBatonTime

    if baton == 0:
        #RGB
        #RED
        GPIO.output(11,GPIO.HIGH)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.LOW)   
	
    elif baton == 2 and (time.time()- batonPassOffTime) < 10:
        #RGB
        #BLUE
        GPIO.output(11,GPIO.LOW)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.HIGH)        
    elif baton == 1 and (time.time()- myBatonTime) < 3:
        #RGB
        #GREEN
        GPIO.output(11,GPIO.LOW)
        GPIO.output(9,GPIO.HIGH)
        GPIO.output(10,GPIO.LOW)
    else:
        #RGB
        GPIO.output(11,GPIO.HIGH)
        GPIO.output(9,GPIO.LOW)
        GPIO.output(10,GPIO.LOW)

try:
    # Connect to the server using the console port argument
    print(lidar.info)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    portNum = int(args.port)
    s.connect(('192.168.4.1', portNum))      
    print ("socket connected to %s" %(portNum))
    s.setblocking(1)
    
    for scan in lidar.iter_scans():
        #Motor Code
        
        widthChecker = 0
        dlStopSign = GPIO.input(26)
        for num in range(len(widthArray)):
            dist = distanceArray[num]
            adjustedWidthThreshold = 2.082*pow(10,-8)*pow(dist,3)-5.32*pow(10,-5)*pow(dist,2)-0.03107*dist+182.8  
            if((widthArray[num] >= adjustedWidthThreshold and widthArray[num] <= 210)):
              
                
                widthChecker = 1
                #print (widthArray[num])
                signSawTime = time.time()
                lastSignDistance = distanceArray[num]
                if distanceArray[num] > 0.8*distanceSeenArray[len(distanceSeenArray)-1] and distanceArray[num] < 1.03*distanceSeenArray[len(distanceSeenArray)-1] or len(distanceSeenArray) == 1: 
 
                    distanceSeenArray = np.append(distanceSeenArray, lastSignDistance)
                    timeArray = np.append(timeArray, signSawTime)
                    print(distanceSeenArray)
                    # Calculating a moving average from the time/distance arrays outputted from DBSCANS
                    # Requires atleast 4 data points to start
                    for i in range(len(distanceSeenArray)-4):
                        if(distanceSeenArray[i] != 0):
                            speed1 = abs((distanceSeenArray[i]-distanceSeenArray[i+1])/(timeArray[i]-timeArray[i+1]))
                            speed2 = abs((distanceSeenArray[i+1]-distanceSeenArray[i+2])/(timeArray[i+1]-timeArray[i+2]))
                            speed3 = abs((distanceSeenArray[i+2]-distanceSeenArray[i+3])/(timeArray[i+2]-timeArray[i+3]))
                            speed4 = abs((distanceSeenArray[i+3]-distanceSeenArray[i+4])/(timeArray[i+3]-timeArray[i+4]))
                            speed5 = abs((distanceSeenArray[i]-distanceSeenArray[i+4])/(timeArray[i]-timeArray[i+4]))
                            avgSpeed = (speed1+speed2+speed3+speed4+speed5)/5
                    
                    actualSignDistance = sqrt(abs(pow(lastSignDistance,2)-pow(558.8,2)))
                    timeUntilIntersection = (actualSignDistance/avgSpeed)
                    print("Time Until Intersection:")
                    print(timeUntilIntersection)               

        leftLineTracker = GPIO.input(5)
        rightLineTracker = GPIO.input(0)
        # This is the shimmying code that keeps the robot along the line.
        if(middleQuadrantScan(scan) == 1) or ((time.time()-obstructionTime) < 2.5):
            forward(0)
        elif(leftLineTracker == 1):
            rightturn()
        elif(rightLineTracker == 1):
            leftturn()
        elif(baton == 1 and ((time.time()-batonPassOffTime) < 10)):
            forward(10)

        else:
            forward(15)
        
        # Starting of the threads
        b = threading.Thread(target = thread1, args = (scan,))
        a = threading.Thread(target = thread2, args = (s,timeUntilIntersection,))
        d = threading.Thread(target = thread3, args = ())
        b.start()
        a.start()
        d.start()
            
# In case of control+c
except KeyboardInterrupt:
    print("Stopping.")
    cleanPorts()
    pwma.stop()
    pwmb.stop()
    lidar.stop()
    lidar.disconnect()
