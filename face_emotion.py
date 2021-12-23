#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import imutils
import numpy as np
import math
from math import pow, asin, sqrt, cos, sin, atan2
from numpy import inf
from numpy import asarray
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from std_msgs.msg import String
import time

from keras.models import load_model
vgg16 = load_model('/home/ubuntu/catkin_ws/src/facial_expression/my_model.h5') # VGGFace network weights
vgg16._make_predict_function()

# Define function to visualize predictions
def visualize_predictions(a):

    pixels = a.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples)
    clases=vgg16.predict(samples)
    c=np.amax(clases)
    d=np.where(clases == c)
    m=int(d[1])
        
    if m == 0:
        n='angry'
    elif m == 1:
        n='disgust'
    elif m == 2:
        n='fear'
    elif m == 3:
        n='happy'
    elif m == 4:
        n='sad'
    elif m == 5:
        n='surprise'
    return n

required_size=(224, 224)
# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('opencv_example', anonymous=True)
# Initialize the CvBridge class
bridge = CvBridge()
BodyClassif = cv2.CascadeClassifier('/home/ubuntu/catkin_ws/src/facial_expression/haarcascade_frontalface_default.xml') # Viola-Jones pre trained classifier

# Define a callback for the Image message
def image_callback(img_msg):
    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))


    if len(cv_image) > 0:
        frame = cv_image
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        Body = BodyClassif.detectMultiScale(gray,1.1,4,minSize=(55,90))

        for (x,y,w,h) in Body:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = frame[y:y+h,x:x+w]
            from PIL import Image
            image = Image.fromarray(rostro)
            image = image.resize(required_size)
            vector1 = asarray(image)
            cv2.putText(frame,str(visualize_predictions(vector1)), (x, y), cv2.FONT_ITALIC, 0.75, (0, 255, 0), 1)

            try:
                pub.publish(bridge.cv2_to_imgmsg(rostro, "bgr8"))

		os.system('cls' if os.name == 'nt' else 'clear')
		from PIL import Image
		image = Image.fromarray(rostro)
		image = image.resize(required_size)
		vector1 = asarray(image)
		print("Emotion detected: ")		
		print(visualize_predictions(vector1))

		emotion_pub = rospy.Publisher('facial_emotion', String, queue_size=10)
		end_time = time.time() + 1
		countTimer = 0
		sleepTime = 0.500
		while time.time() < end_time:
		    emotion_pub.publish(visualize_predictions(vector1))
		    rospy.sleep(0.01)

            except CvBridgeError as e:
                print(e)

    
        cv2.imshow("Image window",frame)
        cv2.waitKey(1)

# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/pepper/camera/front/image_raw", Image, image_callback)
pub = rospy.Publisher('Body', Image)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
  
    rospy.spin()

