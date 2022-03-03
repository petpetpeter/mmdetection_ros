
#!/usr/bin/env python3

import numpy as np
import os
import math
import time
import cv2
import struct
from numpy.core.numeric import NaN
#import open3d as o3d
import matplotlib.pyplot as plt
import glob
import ctypes
import warnings


#ROS Required
import rospy
import ros_numpy
from rospy.core import is_shutdown
from std_msgs.msg import Header,String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Image,CompressedImage
from std_msgs.msg import Bool,Float32MultiArray,Float32
from visualization_msgs.msg import Marker,MarkerArray
#from tomato_detection.srv import SelectTomato,SelectTomatoResponse
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped

#MMD
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

class MMD():
    def __init__(self):
        WEIGHT_PATH     = os.path.dirname(os.path.abspath(__file__))+ "/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        CONFIG_PATH     = os.path.dirname(os.path.abspath(__file__))+ "/configs/faster_rcnn_r50_fpn_1x_coco.py"
        self.model = init_detector(CONFIG_PATH, WEIGHT_PATH, device='cuda:0')
        self.class_names = self.model.CLASSES
        self.bboxDict = dict((name,[]) for name in self.class_names)
        self.result_type = "bbox" # aviable result types: bbox, segm or keypoints[have not been implemented]

    def resetDict(self,bboxDict):
        for k in bboxDict.keys():
            bboxDict[k] = []

    def detect(self, img,conf = 0.8):
        visualize_img = img.copy()
        self.resetDict(self.bboxDict)
        # Prediction
        result = inference_detector(self.model, img)
        visualize_img = self.model.show_result(visualize_img, result, wait_time=0,score_thr=conf)
        for num_class in range(len(self.class_names)):
            if(self.result_type == "bbox"):  #only bounding box detection
                bboxes_by_class = result[num_class] #[x1,y1,x2,y2,score]
            elif(self.result_type == "segm"): #bbox + segm
                bboxes_by_class = result[0][num_class]
                masking_by_class = result[1][num_class]
            #filter lower score than conf
            bboxes_by_class = bboxes_by_class[bboxes_by_class[:,4]>conf]
            #2d array to list of list 
            self.bboxDict[self.class_names[num_class]] = bboxes_by_class.tolist()
            
        return visualize_img


class DepthImageHandler(object):
    def __init__(self):
        self.CAMINFO = {'topic': '/camera/color/camera_info', 'msg': CameraInfo}
        self.COLOR = {'topic': '/camera/color/image_raw', 'msg': Image}
        self.DEPTH = {'topic': '/camera/depth/image_raw', 'msg': Image}
        
        self.H = 720
        self.W = 1280
        self.header = Header() #Use for point cloud publisher

        self.color_image = np.empty((self.H, self.W ,3), dtype=np.uint8)
        self.depth_image = np.empty((self.H, self.W), dtype=np.uint16)
        self.aligned_image  = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask           = np.empty((self.H, self.W), dtype=np.bool)
        self.mask_image     = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask_depth     = np.empty((self.H, self.W), dtype=np.uint8)

        self.camera_matrix = np.array([[0.0, 0, 0.0], [0, 0.0, 0.0], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([0,0,0,0,0], dtype=np.float32)
        self.model = MMD()
        #random rgb in range [0,1] by class from self.model.class_names
        self.marker_colors = np.random.uniform(0,1,[len(self.model.class_names),3])
        self.marker_colors_dict = dict((name,color) for name,color in zip(self.model.class_names,self.marker_colors))

        self.SCORE_THR = 0.85 #  cv2.getTrackbarPos('score','image')
        init_thr= self.SCORE_THR*100
        #create trackbar
        cv2.namedWindow('score_bar')
        cv2.createTrackbar('score','score_bar',1,100,self.scoreCallback)
        cv2.setTrackbarPos('score','score_bar', int (init_thr) )

    def scoreCallback(self, val):
        self.SCORE_THR = val/100.0

    def camInfoCallback(self, msg):
        self.header = msg.header
        self.K = msg.K
        self.width = msg.width  
        self.height = msg.height
        self.ppx = msg.K[2]
        self.ppy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4] 
        
        self.cam_distortion_model = msg.distortion_model
        self.k1 = msg.D[0]
        self.k2 = msg.D[1]
        self.t1 = msg.D[2]
        self.t2 = msg.D[3]
        self.k3 = msg.D[4]
        self.isCamInfo = True
        self.camera_matrix = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([self.k1,self.k2,self.t1,self.t2,self.k3], dtype=np.float32)

    def colorCallback(self, msg):
        self.color_image = ros_numpy.numpify(msg)
        #self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)

    def depthCallback(self, msg):
        #self.depth_image = ros_numpy.numpify(msg).astype(np.uint16) # //RealImage
        numpyImage = ros_numpy.numpify(msg)
        numpyImage = np.nan_to_num(numpyImage, copy=True, nan=0.0)
        self.depth_image = numpyImage
    
    def publishPoint3(self,pos3):
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self.header.frame_id

        point.point.x = pos3[0]
        point.point.y = pos3[1]
        point.point.z = pos3[2]
        
        self.point_pub.publish(point)
    


    def publishMarkerByClass(self,poses_dict:dict,class_name: str,publisher)-> None:
        objectMarkerArray = MarkerArray()
        color = self.marker_colors_dict[class_name]
        id = 0 
        for pos3 in poses_dict[class_name]:
            #print(f"pos3: {pos3} for class: {class_name}") 
            objectMarker = Marker()
            objectMarker.color.r = color[0]
            objectMarker.color.g = color[1]
            objectMarker.color.b = color[2]
            objectMarker.color.a = 1.0
            objectMarker.header.frame_id = self.header.frame_id # Camera Optical Frame
            objectMarker.header.stamp = rospy.Time.now()
            objectMarker.type = 2 # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            # Set the scale of the marker
            objectMarker.scale.x = 0.05
            objectMarker.scale.y = 0.05
            objectMarker.scale.z = 0.05
            # Set the color
            objectMarker.id = id
            objectMarker.pose.position.x = pos3[0]
            objectMarker.pose.position.y = pos3[1]
            objectMarker.pose.position.z = pos3[2]
            objectMarker.lifetime = rospy.Duration(1)
            objectMarkerArray.markers.append(objectMarker)
            id += 1
        publisher.publish(objectMarkerArray)
        
    
    def publishImage(self,image):
        msg = ros_numpy.msgify(Image, image,encoding = "bgr8")
        # Publish new image
        self.image_pub.publish(msg)
    

    def pixelCrop(self,img, dim,pixel):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(pixel[0]), int(pixel[1])
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
        
    def depthPixelToPoint3(self, depth_image, U, V):
        V =  np.clip(V,0,self.height-1)
        U =  np.clip(U,0,self.width-1)  
        
        x = (U - self.K[2])/self.K[0]
        y = (V - self.K[5])/self.K[4]     
        nearPixel = self.pixelCrop(depth_image,[5,5],(U,V))
        meanDepth = np.mean(nearPixel[np.nonzero(nearPixel)]) #Mean of non zero in neighborhood pixel
        if np.isnan(meanDepth):
            rospy.logwarn("Nan depth value, 0 mean depth is returned")
            meanDepth = 0
        #print(meanDepth,"mean")
        z = meanDepth # /1000 for real camera
        x *= z
        y *= z
        # print(U,V,x,y,z)
        point3 = [x, y, z]
        return point3
    
    def pos3FromBboxes(self, depthImage, bboxes:list)->list:
        pos3List = []
        #print(f"bboxes: {bboxes} in pos3FromBboxes")
        for (x1,y1,x2,y2,score) in bboxes:
            cX,cY = (int((x2+x1)/2),int((y2+y1)/2))
            pos3List.append(self.depthPixelToPoint3(depthImage,cX,cY))
        return pos3List

    def process(self):
        t1 = time.time()
        color_image, depth_image    = self.color_image, self.depth_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        # color_image = cv2.imread("inference/images/tomato.jpeg")
        #print(self.SCORE_THR)
        detected_image = self.model.detect(color_image,conf=self.SCORE_THR)
        current_results_dict = self.model.bboxDict
        #print(current_results_dict)
        camPosDict = {}
        for name in self.model.class_names:
            if len(current_results_dict[name]) > 0: # if there is at least one bbox
                camPosDict[name] = self.pos3FromBboxes(depth_image,current_results_dict[name])
                self.publishMarkerByClass(camPosDict,name,self.class_pub[name])
        
        self.publishImage(detected_image)
        cv2.imshow('frame', detected_image)
        cv2.waitKey(1)

    def rosinit(self):
        rospy.init_node('markerFinder', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.CAMINFO['topic'], self.CAMINFO['msg'], self.camInfoCallback)
        rospy.Subscriber(self.COLOR['topic'], self.COLOR['msg'], self.colorCallback)
        rospy.Subscriber(self.DEPTH['topic'], self.DEPTH['msg'], self.depthCallback)
        #rospy.Subscriber(self.PC['topic'],    self.PC['msg'],    self.pcCallback)
    
        ###publisher        
        self.image_pub = rospy.Publisher('/detectedImage', Image, queue_size=1)
        #make a publisher dict for each class
        self.class_pub = {}
        for name in self.model.class_names:
            self.class_pub[name] = rospy.Publisher(f'/object/{name}', MarkerArray, queue_size=10)
                    
        rospy.wait_for_message(self.CAMINFO['topic'], self.CAMINFO['msg'])            
        while not rospy.is_shutdown():
            self.process()

if __name__ == '__main__':
    try:
        _depthImageHandler = DepthImageHandler()
        _depthImageHandler.rosinit()

    except rospy.ROSInterruptException:
        pass