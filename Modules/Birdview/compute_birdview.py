import sys, os

sys.path.append("../..")

import numpy as np
import cv2
from Utils.Config.app import getConfig
import math
import itertools
import matplotlib.pyplot as plt

import time

import glob

sys.path.append("../..")
from Utils.Cv2_Effect.app import *

COLOR_RED =     (60, 30, 250)
COLOR_GREEN =   (0, 255, 0)
COLOR_ORANGE =  (0, 165, 255)

COLOR_BLUE = (255, 0, 0)
COLOR_BLUE_INNER = (142, 88, 17)

COLOR_BLUE = (255, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_BLUE_INNER = (142, 88, 17)

COLOR_ARROW = (234,118,39)

COLOR_1 =   (128, 193, 152)
COLOR_2 =   (57, 167, 163)
COLOR_3 =   (147, 110, 23)
COLOR_4 =   (114, 110, 23)
COLOR_5 =   (76, 110, 23)
COLOR_6 =   (45, 110, 23)
COLOR_7 =   (52, 100, 77)
COLOR_8 =   (35, 100, 77)
COLOR_9 =   (85, 67, 77)
COLOR_10 =  (21, 25, 77)
COLOR_11 =  (52, 27, 36)
COLOR_12 =  (70, 95, 142)

COLOR_ARRAY = [ COLOR_1,
                COLOR_2,
                COLOR_3,
                COLOR_4,
                COLOR_5,
                COLOR_6,
                COLOR_7,
                COLOR_8,
                COLOR_9,
                COLOR_10,
                COLOR_11,
                COLOR_12                
                ]


BIG_CIRCLE = 70
SMALL_CIRCLE = 30

VELOCITY_THRESHOLD = 5

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class BirdView(object):
    def __init__(self):
        self.iconSize =  (42, 78)
        self.halfIconSize = (self.iconSize[0]//2, self.iconSize[1]//2)

        self.preOutputTrackingBboxes = []
        self.previousBlinked = 0
        self.blinkTime = 3
        self.textWarning = False
        self.count = -1
        self.blinked = False

        self.socialDistanceTimerStart = 0
        self.socialDistanceTimerEnd = 0
        self.isWarningBefore = False

        # Load black and red human
        blackHuman = cv2.imread("Resources/BirdView_Icons/black_human.jpg")
        redHuman = cv2.imread("Resources/BirdView_Icons/red_human.jpg")
        self.blackHuman = cv2.resize(blackHuman, self.iconSize)
        self.redHuman = cv2.resize(redHuman, self.iconSize)
        
        self.read_R_T_matrix()
        # self.PreNosePoints = []

        # Load stand people images
        imgPaths = glob.glob(os.path.join("Resources", "Stand_People_Images", "*.jpeg"))
        self.peopleImgs = []
        for imgPath in imgPaths:
            img = cv2.imread(imgPath)
            self.peopleImgs.append(img)
        
    def read_R_T_matrix(self):
        
        config = getConfig("Modules/Birdview/config.ini")

        self.imageMode = True if int(config['SHOW_ARROW']['PEOPLE_ICON'])==1 else False

        self.ORG_IMAGE_WIDTH        = int(config['CALIB']['ORG_IMAGE_WIDTH'])
        self.ORG_IMAGE_HEIGHT       = int(config['CALIB']['ORG_IMAGE_HEIGHT'])

        # Minimum distance to warning social distance
        self.distance_minimum       = int(config['SOCIAL_DISTANCE_DETECT']['MINIMUM_DISTANCE'])

        # Social distance detection, need set to true or false
        self.socialDistanceDetect   = int(config['SOCIAL_DISTANCE_DETECT']['ENABLED'])
        
        # Bird view image size
        self.birdview_width         = int(config["CALIB"]["BIRDVIEW_IMAGE_WIDTH"])
        self.birdview_height        = int(config["CALIB"]["BIRDVIEW_IMAGE_HEIGHT"])

        # Show arrow
        self.normalizeFactor        = int(config["SHOW_ARROW"]["NORMALIZE_FACTOR"])

        # Social distance time
        self.socialDistanceTime             = int(config['SOCIAL_DISTANCE_TIME']['SOCIAL_DISTANCE_TIME'])
        self.greenTime                      = int(config['SOCIAL_DISTANCE_TIME']['GREEN_TIME'])
        self.orangeTime                     = int(config['SOCIAL_DISTANCE_TIME']['ORANGE_TIME'])
        self.redTime                        = int(config['SOCIAL_DISTANCE_TIME']['RED_TIME'])


        R11 = float(config["TMATRIX_O"]["R11"])
        R12 = float(config["TMATRIX_O"]["R12"]) 
        R13 = float(config["TMATRIX_O"]["R13"])
        R21 = float(config["TMATRIX_O"]["R21"])
        R22 = float(config["TMATRIX_O"]["R22"])
        R23 = float(config["TMATRIX_O"]["R23"])
        R31 = float(config["TMATRIX_O"]["R31"])
        R32 = float(config["TMATRIX_O"]["R32"])
        R33 = float(config["TMATRIX_O"]["R33"])
        TX  = float(config["TMATRIX_O"]["TX"])
        TY  = float(config["TMATRIX_O"]["TY"])
        TZ  = float(config["TMATRIX_O"]["TZ"])

        self.Tmatrix_o = np.array([     [R11, R12, R13, TX],
                                        [R21, R22, R23, TY],
                                        [R31, R32, R33, TZ],
                                        [0.,  0.,  0.,  1.,]])

        
        # Camera matrix
        self.fx = float(config["CAMERA_MATRIX"]["FX"])
        self.fy = float(config["CAMERA_MATRIX"]["FY"])
        self.cx = float(config["CAMERA_MATRIX"]["CX"])
        self.cy = float(config["CAMERA_MATRIX"]["CY"])
        
        camera_mtx =    [[self.fx, 0.0, self.cx], 
                        [0.0, self.fy, self.cy], 
                        [0.0, 0.0, 1]]
        camera_mtx = np.asarray(camera_mtx, dtype=np.float)
        camera_mtx = np.reshape(camera_mtx, (3, 3))
        
        # aruco marker size
        self.aruco_marker_size = float(config["CALIB"]["ARUCO_MARKER_SIZE"])
        
        # OFFSET 
        self.resolution_factor = float(config["CALIB"]["RESOLUTION_FACTOR"])
        
    
    def warpPerspective(self, image_input):
        # perimage = image_input.copy()
        birdview_image = cv2.warpPerspective(image_input, self.M, (self.birdview_width, self.birdview_height))
        return birdview_image
    

    def get_centroids_and_groundpoints(self, array_boxes_detected):
        """
        For every bounding box, compute the centroid and the point located on the bottom center of the box
        @ array_boxes_detected : list containing all our bounding boxes 
        """

        # Initialize empty centroid and ground point lists 
        array_centroids,array_groundpoints = [],[]

        for index,box in enumerate(np.array(array_boxes_detected[0:4])):
            # Draw the bounding box 
            # Get the both important points
            centroid,ground_point = self.get_points_from_box(box)
            array_centroids.append(centroid)
            array_groundpoints.append(centroid)

        return array_centroids, array_groundpoints


    def get_points_from_box(self, box):
        """
        Get the center of the bounding and the point "on the ground"
        @ param = box : 2 points representing the bounding box
        @ return = centroid (x1,y1) and ground point (x2,y2)
        """
        # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
        center_x = int(((box[1]+box[3])/2))
        center_y = int(((box[0]+box[2])/2))
        # Coordiniate on the point at the bottom center of the box
        center_y_ground = center_y + ((box[2] - box[0])/2)
        return (center_x,center_y),(center_x,int(center_y_ground))


    def change_color_on_topview_pair(self, pair, bird_view_img):
        """
        Draw red circles for the designated pair of points 
        """
        cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
        cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)

        cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
        cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)
    
    def compute_point_perspective_transformation(self, matrix,list_downoids):
        """ Apply the perspective transformation to every ground point which have been detected on the main frame.
        @ matrix : the 3x3 matrix 
        @ list_downoids : list that contains the points to transform
        return : list containing all the new points
        """
        # Compute the new coordinates of our points
        print("list_downoids ====== ",list_downoids)
        list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
        # Loop over the points and add them to the list that will be returned
        transformed_points_list = list()
        for i in range(0, transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
        return transformed_points_list

    def change_color_on_topview(self, birdview_image, pair):
        """
        Draw red circles for the designated pair of points 
        """
        # cv2.circle(birdview_image, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)

        # cv2.circle(birdview_image, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)

        if birdview_image[pair[0][1] - self.halfIconSize[1] : pair[0][1] + self.halfIconSize[1], \
            pair[0][0] - self.halfIconSize[0] : pair[0][0] + self.halfIconSize[0], :].shape != \
            self.redHuman.shape and \
            birdview_image[pair[1][1] - self.halfIconSize[1] : pair[1][1] + self.halfIconSize[1], \
            pair[1][0] - self.halfIconSize[0] : pair[1][0] + self.halfIconSize[0], :].shape != \
            self.redHuman.shape:
            return

        birdview_image[pair[0][1] - self.halfIconSize[1] : pair[0][1] + self.halfIconSize[1], pair[0][0] - self.halfIconSize[0] : pair[0][0] + self.halfIconSize[0], :]= self.redHuman

        birdview_image[pair[1][1] - self.halfIconSize[1] : pair[1][1] + self.halfIconSize[1], pair[1][0] - self.halfIconSize[0] : pair[1][0] + self.halfIconSize[0], :]= self.redHuman
    

    def show_people_on_topview(self, orgImg, birdview_image, outputTrackingBboxes):

        # Both of our lists that will contain the centroids coordonates and the ground points
        if len(outputTrackingBboxes) > 0:
            array_centroids, array_groundpoints = self.get_centroids_and_groundpoints(outputTrackingBboxes[0:4])

            # Use the transform matrix to get the transformed coordonates
            transformed_downoids = self.compute_point_perspective_transformation(self.transformed_matrix, array_groundpoints)

            # Show every point on the top view image 
            for point in transformed_downoids:
                x,y = point
                cv2.circle(birdview_image, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(birdview_image, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)

    
    def show_topview(self, outputTrackingBboxes):
        
        birdview_image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        
        if not self.imageMode:
            for prePerson in self.preOutputTrackingBboxes:
                for person in outputTrackingBboxes:

                    if prePerson[4] == person[4]:
                        
                        # ID
                        id = person[4]
                        colorIdx = id % 12

                        # Now person
                        x0_now, y0_now, x1_now, y1_now = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                        center_now = ((x0_now+x1_now)//2,(y0_now+y1_now)//2)
                        # Last Person
                        x0_last, y0_last, x1_last, y1_last = int(prePerson[0]),int(prePerson[1]),int(prePerson[2]),int(prePerson[3])
                        center_last = ((x0_last+x1_last)//2,(y0_last+y1_last)//2)

                        # Draw centroid point matching with people
                        
                        # cv2.circle(birdview_image,center_now, SMALL_CIRCLE - 2, COLOR_BLUE_INNER, -1)
                        cv2.circle(birdview_image,center_now, SMALL_CIRCLE, COLOR_ARRAY[colorIdx], -1)

                        # Calculate arrow based on vector between last_point and now_point
                        startPoint = center_now

                        ABdistance = math.sqrt((center_now[0]-center_last[0])**2 + (center_now[1]-center_last[1])**2)
                        if ABdistance <= VELOCITY_THRESHOLD:
                            break
                        
                        # xC = self.normalizeFactor*((center_now[0]-center_last[0])/ABdistance + center_last[0])
                        # yC = self.normalizeFactor*((center_now[1]-center_last[1])/ABdistance + center_last[1])
                        # endPoint = (xC, yC)

                        vector_v = np.array([center_now[0] - center_last[0], center_now[1] - center_last[1]])
                        vector_u = BIG_CIRCLE/np.linalg.norm(vector_v)*vector_v    

                        endPoint = (center_now[0] + vector_u[0], center_now[1] + vector_u[1])

                        # draw velocity arrow line
                        cv2.arrowedLine(birdview_image, (int(startPoint[0]), int(startPoint[1])), (int(endPoint[0]), int(endPoint[1])), COLOR_ARROW, thickness = 4, tipLength=0.6)
        else:
            for prePerson in self.preOutputTrackingBboxes:
                for person in outputTrackingBboxes:

                    if prePerson[4] == person[4]:
                        
                        # ID
                        id = person[4]
                        colorIdx = id % 12

                        # Now person
                        x0_now, y0_now, x1_now, y1_now = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                        center_now = ((x0_now+x1_now)//2,(y0_now+y1_now)//2)
                        # Last Person
                        x0_last, y0_last, x1_last, y1_last = int(prePerson[0]),int(prePerson[1]),int(prePerson[2]),int(prePerson[3])
                        center_last = ((x0_last+x1_last)//2,(y0_last+y1_last)//2)

                        # Draw centroid point matching with people                
                        # cv2.circle(birdview_image,center_now, SMALL_CIRCLE - 2, COLOR_BLUE_INNER, -1)
                        # cv2.circle(birdview_image, center_now, SMALL_CIRCLE, COLOR_ARRAY[colorIdx], -1)

                        # Compute alpha angle
                        myradians = math.atan2(center_last[1]-center_now[1], center_last[0]-center_now[0])
                        alphaAngle = math.degrees(myradians)
                        # Compute distance to ignore small velocity
                        ABdistance = math.sqrt((center_now[0]-center_last[0])**2 + (center_now[1]-center_last[1])**2)
                        
                        

                        # Resize people image
                        peopleImg = cv2.resize(self.peopleImgs[0], (70, 70))
                        # Merge people image into birdview image
                        imgH, imgW, _ = peopleImg.shape
                        startX = center_now[0] - (imgW//2)
                        startY = center_now[1] - (imgH//2)
                        endX = center_now[0] + (imgW//2)
                        endY = center_now[1] + (imgH//2)
                        # Ignore bad results
                        if startX < 0 or startY < 0 or endX>=1280 or endY>=720:
                            continue
                        # Rotate people image if velocity is enough VELOCITY_THRESHOLD
                        if ABdistance > VELOCITY_THRESHOLD:
                            peopleImg = rotateImage(peopleImg, alphaAngle)
                        birdview_image[startY:endY, startX:endX, :] = peopleImg

                        if ABdistance <= VELOCITY_THRESHOLD:
                            break

                        # Calculate arrow based on vector between last_point and now_point
                        startPoint = center_now
                        
                        # xC = self.normalizeFactor*((center_now[0]-center_last[0])/ABdistance + center_last[0])
                        # yC = self.normalizeFactor*((center_now[1]-center_last[1])/ABdistance + center_last[1])
                        # endPoint = (xC, yC)

                        vector_v = np.array([center_now[0] - center_last[0], center_now[1] - center_last[1]])
                        vector_u = BIG_CIRCLE/np.linalg.norm(vector_v)*vector_v    

                        endPoint = (center_now[0] + vector_u[0], center_now[1] + vector_u[1])

                        # draw velocity arrow line
                        cv2.arrowedLine(birdview_image, (int(startPoint[0]), int(startPoint[1])), (int(endPoint[0]), int(endPoint[1])), COLOR_ARROW, thickness = 4, tipLength=0.6)
                                
        # Update Bounding box
        self.preOutputTrackingBboxes = outputTrackingBboxes

        # Social distance show
        if self.socialDistanceDetect == 1:
            centers = []
            for person in outputTrackingBboxes:
                x0_now, y0_now, x1_now, y1_now = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                centers.append( ((x0_now+x1_now)//2,(y0_now+y1_now)//2) )
            
            # Check if 2 or more people have been detected (otherwise no need to detect)
            if len(centers) >= 2:
                # Iterate over every possible 2 by 2 between the points combinations 
                for i,pair in enumerate(itertools.combinations(centers, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(self.distance_minimum):
                        # Change the colors of the points that are too close from each other to red
                        # if not (pair[0][0] > self.birdview_width or pair[0][0] < 0 or pair[0][1] > self.birdview_height  
                        # or pair[0][1] < 0 or pair[1][0] > self.birdview_width or pair[1][0] < 0 or pair[1][1] > self.birdview_width  or pair[1][1] < 0):
                        # self.change_color_on_topview(birdview_image, pair)
                        pass
        return birdview_image

    def xyd_to_3d(self, x, y, d, depth_scale):
        d *= depth_scale
        x = (x - self.cx) * d / self.fx
        y = (y - self.cy) * d / self.fy
        p3d = np.array([x, y, d])
        return p3d
      
    def convert_centroid2birdview(self, orgImg, outputTrackingBboxes, depth_image, depth_scale):

        # Blink interval is 4 frames (NUM_SHOW_WARNING % BLINK_TIME == 0)
        BLINK_TIME = 4
        NUM_SHOW_WARNING = 20 + (BLINK_TIME-1)
        TIME_RESET = 60

        centers = []

        birdview_image = np.full((720, 1280, 3), 255, dtype=np.uint8)

        # plt.imshow(depth_image)
        # plt.show()
        socialDistanceDetected = False
        for prePerson in self.preOutputTrackingBboxes:
            for person in outputTrackingBboxes:
                
                # Compare has same id
                if prePerson[4] == person[4]:

                    socialDistanceDetected = True
                    
                    # Current person
                    x0_now, y0_now, x1_now, y1_now = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                    xNow = (x0_now+x1_now)//2
                    yNow = (y0_now+y1_now)//2

                    # Previous point
                    x0_last, y0_last, x1_last, y1_last = int(prePerson[0]),int(prePerson[1]),int(prePerson[2]),int(prePerson[3])
                    xLast = (x0_last+x1_last)//2
                    yLast = (y0_last+y1_last)//2

                    # To debug depth in centroid point
                    # cv2.putxTet(orgImg, str(depth_image[yNow, xNow]),(xNow, yNow), cv2.FONT_HERSHEY_COMPLEX, 1, (60,30,250), 3)

                    # Ignore invalid depth
                    if float(depth_image[yNow, xNow]) == 0:
                        continue

                    # Transform coordinator now
                    X, Y, Z = self.xyd_to_3d(xNow, yNow, float(depth_image[yNow, xNow]), depth_scale)
                    # cv2.circle(orgImg, (xNow, yNow), 5, COLOR_RED, -1)
                    
                    # cv2.imshow("test", orgImg)
                    # cv2.waitKey(0)
                    # print("xNow, yNow, depth === ", xNow, yNow, float(depth_image[yNow, xNow]))
                    # print("============= before =============")
                    # print(X, Y, Z)

                    transpose_point = np.expand_dims(np.array([X, Y, Z, 1]), axis=0).T
                    transformedPoint = np.matmul(self.Tmatrix_o, transpose_point)

                    # print("============= After =============")
                    # print(transformedPoint)

                    transformedPoint = np.squeeze(transformedPoint)
                    xTransformed = int(640 + self.resolution_factor * transformedPoint[0])
                    yTransformed = int(360 - self.resolution_factor * transformedPoint[1])

                    # ID

                    # id = person[4]
                    # colorIdx = id % 12

                    # cv2.circle(birdview_image, (xTransformed, yTransformed), SMALL_CIRCLE, COLOR_ARRAY[colorIdx], -1)
                    
                    # if (yTransformed + self.halfIconSize[1] - yTransformed + self.halfIconSize[1] < self.blackHuman.shape[0]) \
                    #     or xTransformed + self.halfIconSize[0] -  xTransformed + self.halfIconSize[0] < self.blackHuman.shape[1]:
                    #     continue
                    # print(yTransformed - self.halfIconSize[1], yTransformed + self.halfIconSize[1])
                    # print(birdview_image.shape)
                    if birdview_image[yTransformed - self.halfIconSize[1] : yTransformed + self.halfIconSize[1], \
                         xTransformed - self.halfIconSize[0] : xTransformed + self.halfIconSize[0], :].shape != \
                             self.blackHuman.shape:
                        continue
                    # exit()
                    birdview_image[yTransformed - self.halfIconSize[1] : yTransformed + self.halfIconSize[1], xTransformed - self.halfIconSize[0] : xTransformed + self.halfIconSize[0], :]= self.blackHuman

                    centers.append((xTransformed, yTransformed))

                    # Transform coordinator previous
                    X, Y, Z = self.xyd_to_3d(xLast, yLast, float(depth_image[yLast, xLast]), depth_scale)
                    transpose_point = np.expand_dims(np.array([X, Y, Z, 1]), axis=0).T
                    transformedPoint = np.matmul(self.Tmatrix_o, transpose_point)
                    transformedPoint = np.squeeze(transformedPoint)
                    
                    preXTransformed = int(640 + self.resolution_factor * transformedPoint[0])
                    preYTransformed = int(360 - self.resolution_factor * transformedPoint[1])

                    if preXTransformed <= 0 or preYTransformed <= 0 or preXTransformed >= 1280 or preYTransformed >= 720:
                        continue

                    # Compute distance
                    ABdistance = math.sqrt((xTransformed-preXTransformed)**2 + (yTransformed-preYTransformed)**2)
                    
                    if ABdistance <= VELOCITY_THRESHOLD:
                        break

                    # Compute velocity vector
                    vector_v = np.array([xTransformed - preXTransformed, yTransformed - preYTransformed])
                    vector_u = BIG_CIRCLE/np.linalg.norm(vector_v)*vector_v    
                    endPoint = (xTransformed + vector_u[0], yTransformed + vector_u[1])

                    # draw velocity arrow line
                    cv2.arrowedLine(birdview_image, (xTransformed, yTransformed), (int(endPoint[0]), int(endPoint[1])), COLOR_ARROW, thickness = 4, tipLength=0.6)

        # Reset social distance start time
        if socialDistanceDetected == False and self.socialDistanceTimerEnd - self.socialDistanceTimerStart > TIME_RESET:
            self.socialDistanceTimerStart = 0
            self.count = -1
        
        # Update Points
        self.preOutputTrackingBboxes = outputTrackingBboxes

        # Social distance show
        if self.socialDistanceDetect == 1:
            
            # Check if 2 or more people have been detected (otherwise no need to detect)
            if len(centers) >= 2:
                # Iterate over every possible 2 by 2 between the points combinations 
                for i,pair in enumerate(itertools.combinations(centers, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(self.distance_minimum):
                        
                        # if not (pair[0][0] > self.birdview_width or pair[0][0] < 0 or pair[0][1] > self.birdview_height  
                        # or pair[0][1] < 0 or pair[1][0] > self.birdview_width or pair[1][0] < 0 or pair[1][1] > self.birdview_width  or pair[1][1] < 0):
                        
                        # Change the colors of the points that are too close from each other to red
                        self.change_color_on_topview(birdview_image, pair)

                        if self.socialDistanceTimerStart == 0:
                            self.socialDistanceTimerStart = time.time()
                        
                        self.socialDistanceTimerEnd = time.time()
                        if (self.socialDistanceTimerEnd - self.socialDistanceTimerStart > self.socialDistanceTime) and ((self.count + 1 > BLINK_TIME)\
                        or (self.count == -1)):
                            self.count = 0

                        # cv2.putText(orgImg, str(socialDistanceTimerEnd - self.socialDistanceTimerStart),(int(orgImg.shape[1] *0.5), int(orgImg.shape[0]*0.5)), cv2.FONT_HERSHEY_COMPLEX, 1, (60,30,250), 3)

    
                # cv2.putText(orgImg, str(self.count),(int(orgImg.shape[1] *0.5), int(orgImg.shape[0]*0.6)), cv2.FONT_HERSHEY_COMPLEX, 1, (60,30,250), 3)
                if 0 <= self.count <= NUM_SHOW_WARNING:
                    self.count += 1
                    if self.count % BLINK_TIME == 0 and self.count != 0:
                        self.blinked = not self.blinked 
                    if self.blinked:
                        (label_width, label_height), baseline = cv2.getTextSize("Social Distance Warning", cv2.FONT_HERSHEY_COMPLEX, 1, 3)
                        startX = orgImg.shape[1]//2 - label_width//2 - 20
                        startY = int(orgImg.shape[0]*0.1) - label_height//2 - 30
                        endX   = startX + label_width  + 20
                        endY   = startY + label_height + 40
                        orgImg = rounded_rectangle(orgImg, (startX, startY),(endY, endX), color= (51, 51, 51), radius=0.7, thickness=-1)
                        
                        if (self.socialDistanceTimerEnd - self.socialDistanceTimerStart > self.redTime):
                            cv2.putText(orgImg, "Social Distance Warning",(orgImg.shape[1]//2 - label_width//2, int(orgImg.shape[0]*0.1)), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_RED, 3)
                        elif (self.socialDistanceTimerEnd - self.socialDistanceTimerStart > self.orangeTime):
                            cv2.putText(orgImg, "Social Distance Warning",(orgImg.shape[1]//2 - label_width//2, int(orgImg.shape[0]*0.1)), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_ORANGE, 3)
                        elif (self.socialDistanceTimerEnd - self.socialDistanceTimerStart > self.greenTime):
                            cv2.putText(orgImg, "Social Distance Warning",(orgImg.shape[1]//2 - label_width//2, int(orgImg.shape[0]*0.1)), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_GREEN, 3)

                    # if self.count >= NUM_SHOW_WARNING:
                    #     self.socialDistanceTimerStart = 0
                    #     self.count = -1


        return birdview_image
      