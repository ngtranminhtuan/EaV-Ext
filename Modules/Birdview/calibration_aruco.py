import sys, os

sys.path.append("../..")

import numpy as np
import cv2
import cv2.aruco as aruco
from Utils.Config.app import getConfig
import logging
import math
from realsense_camera import RealsenseCamera

fx = 638.7794189453125
fy = 638.1564331054688
cx = 627.7446899414062
cy = 362.18048095703125


def xyd_to_3d(x, y, d, depth_scale):

    d *= depth_scale
    x = (x - cx) * d / fx
    y = (y - cy) * d / fy
    p3d = np.array([x, y, d])
    return p3d


if __name__ == "__main__":
    # get configuration file
    config = getConfig()
    
    # Init and read RealSense Camera Params
    camera = RealsenseCamera()

    # get RGB image and depth
    # Connect to Camera
    logging.info('[INFO] Connecting to the camera...')
    camera.connect()
    
    img_h = camera.intrinsic_paramas_.height
    img_w = camera.intrinsic_paramas_.width
    
    # Intrinsic params
    fx = camera.intrinsic_paramas_.fx
    fy = camera.intrinsic_paramas_.fy
    cx = camera.intrinsic_paramas_.ppx
    cy = camera.intrinsic_paramas_.ppy
    
    # Camera matrix
    camera_mtx = [[fx, 0.0, cx], 
                  [0.0, fy, cy], 
                  [0.0, 0.0, 1]]
    camera_mtx = np.asarray(camera_mtx, dtype=np.float)
    camera_mtx = np.reshape(camera_mtx, (3, 3))
    
    dist_coff = camera.intrinsic_paramas_.coeffs
    dist_coff = np.asarray(dist_coff, dtype=np.float)
    dist_coff = np.reshape(dist_coff, (1, 5))
    
    # aruco marker size
    aruco_marker_size = float(config["CALIB"]["ARUCO_MARKER_SIZE"])
    
    # OFFSET 
    offset = float(config["CALIB"]["OFFSET"])

    # Depth scale
    depth_scale = camera.get_depth_scale()

    # --------------------------- ARUCO TRACKER ---------------------------
    while True:
        # Read images
        images              = camera.get_images_bundle()
        # RGB color image
        color_image         = images["color_image"]
        color_image         = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # depth image
        depth_image         = images["depth_image"]
        depth_image_color   = images["depth_image_color"]

        # convert BGR image to gray image
        gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 3

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
        # print("[INFO] IDs of Aruco: ", ids)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        birdview_image = np.zeros((1280, 720), dtype=float)

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec , _ = aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, camera_mtx, dist_coff)
            conner_arr = np.asarray(corners)

            for i in range(0, ids.size):
                print("========================================================================")
                print("[INFO] Id: ", ids[i])

                # draw axis for the aruco markers
                aruco.drawAxis(color_image, camera_mtx, dist_coff, rvec[i], tvec[i], 0.1)
                
                centerX = (conner_arr[i][0][0][0] + conner_arr[i][0][1][0] + conner_arr[i][0][2][0] + conner_arr[i][0][3][0]) / 4
                centerY = (conner_arr[i][0][0][1] + conner_arr[i][0][1][1] + conner_arr[i][0][2][1] + conner_arr[i][0][3][1]) / 4
                center = (int(centerX), int(centerY))                
                center_depth = (float)(depth_image[int(centerY), int(centerX)]*depth_scale)
                
                X,Y,Z = xyd_to_3d(centerX, centerY, (float)(depth_image[int(centerY), int(centerX)]), depth_scale)
                
                # print("============ X,Y,Z =============")
                # print(X,Y,Z)
                
                center_c = np.array([X, Y, Z, 1.0], dtype=np.float)
                center_c = np.reshape(center_c, (4, 1))
                
                # tvec[i][0][0] = X
                # tvec[i][0][1] = Y
                # tvec[i][0][2] = Z
                
                Tmatrix_c = np.zeros((4, 4), dtype=np.float)
                rotation_matrix = np.matrix(cv2.Rodrigues(rvec[i])[0])
                rotation_matrix = np.reshape(rotation_matrix, (3, 3))
                Tmatrix_c[0:3, 0:3] = rotation_matrix
                Tmatrix_c[0:3, 3:] = tvec[i].T
                Tmatrix_c[3:, 3:] = 1.0
                Tmatrix_c = np.reshape(Tmatrix_c, (4, 4))
                
                # print("============ Tmatrix_c =============")
                # print(Tmatrix_c)

                Tmatrix_o = np.zeros((4, 4), dtype=np.float)
                Tmatrix_o = np.linalg.inv(Tmatrix_c)
                # Tmatrix_o[2, 3:] = center_depth
                Tmatrix_o = np.reshape(Tmatrix_o, (4, 4))
                
                print("============ Tmatrix_o =============")
                print(Tmatrix_o)
                
                center_o = np.matmul(Tmatrix_o, center_c)
                
                print("============ center_o =============")
                print(center_o)
                
                ##################
                # temp_depth = (float)(depth_image[int(conner_arr[i][0][1][1]), 
                #                                          int(conner_arr[i][0][1][0]) ])
                # X1,Y1,Z1 = xyd_to_3d(int(conner_arr[i][0][1][0]), 
                #                      int(conner_arr[i][0][1][1]), 
                #                     temp_depth, depth_scale)
                
                # p1_c = np.array([X1, Y1, Z1, 1.0], dtype=np.float)
                # p1_c = np.reshape(p1_c, (4, 1))
                # p1_o = np.matmul(Tmatrix_o, p1_c)
                # print("===========  p1_o  ==============")
                # print(p1_o)
                # print(X1, Y1, Z1)
                # print(Tmatrix_o)
                # print(temp_depth)
                
                # ##################
                # temp_depth = (float)(depth_image[int(conner_arr[i][0][2][1]), 
                #                                          int(conner_arr[i][0][2][0])])
                # X2, Y2, Z2 = xyd_to_3d(int(conner_arr[i][0][2][0]), 
                #                      int(conner_arr[i][0][2][1]), 
                #                      temp_depth, depth_scale)
                
                # p2_c = np.array([X2, Y2, Z2, 1.0], dtype=np.float)
                # p2_c = np.reshape(p2_c, (4, 1))
                # p2_o = np.matmul(Tmatrix_o, p2_c)
                # print("===========  p2_c  ==============")
                # print(p2_o)
                # print(X2, Y2, Z2)
                # print(Tmatrix_o)
                # print(temp_depth)
                
                # ##################
                # temp_depth =  (float)(depth_image[int(conner_arr[i][0][3][1]), 
                #                                          int(conner_arr[i][0][3][0])])
                # X3, Y3, Z3 = xyd_to_3d(int(conner_arr[i][0][3][0]), 
                #                      int(conner_arr[i][0][3][1]), 
                #                    temp_depth, depth_scale)
                
                # p3_c = np.array([X3, Y3, Z3, 1.0], dtype=np.float)
                # p3_c = np.reshape(p3_c, (4, 1))
                # p3_o = np.matmul(Tmatrix_o, p3_c)
                # print("===========  p3_c  ==============")
                # print(p3_o)
                # print(X3, Y3, Z3)
                # print(Tmatrix_o)
                # print(temp_depth)


            # draw a square around the markers
            aruco.drawDetectedMarkers(color_image, corners)
            
            pers_image = color_image.copy()
            translation_vec = tvec[i].squeeze()
            rotation_vec    = rvec[i].squeeze()
            pts_3D = np.asarray([[0, 0, translation_vec[-1]],
                                 [aruco_marker_size + offset, 0, translation_vec[-1]],
                                 [aruco_marker_size + offset, aruco_marker_size + offset, translation_vec[-1]],
                                [0, aruco_marker_size + offset, translation_vec[-1]]], dtype=np.float32)
            
            dst_pts = camera_mtx @ pts_3D.transpose()
            dst_pts = dst_pts.transpose()[:, :2] / translation_vec[-1]
            M = cv2.getPerspectiveTransform(corners[0].astype(np.float32), dst_pts.astype(np.float32))
            birdview_image = cv2.warpPerspective(pers_image, M, (1280, 720))
            
        cv2.imshow("color_frame", color_image)
        cv2.imshow("bird_view", birdview_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    # camera.release()
    cv2.destroyAllWindows()

    # write to output
    cv2.imwrite("bird_view.png", birdview_image)
    
    config.set("RVEC", "ROLL", str(rotation_vec[0]))
    config.set("RVEC", "PITCH", str(rotation_vec[1]))
    config.set("RVEC", "YAW", str(rotation_vec[2]))
    
    config.set("TVEC", "X", str(translation_vec[0]))
    config.set("TVEC", "Y", str(translation_vec[1]))
    config.set("TVEC", "Z", str(translation_vec[2]))

    config.set("TMATRIX_O", "R11", str(Tmatrix_o[0][0]))
    config.set("TMATRIX_O", "R12", str(Tmatrix_o[0][1]))
    config.set("TMATRIX_O", "R13", str(Tmatrix_o[0][2]))

    config.set("TMATRIX_O", "R21", str(Tmatrix_o[1][0]))
    config.set("TMATRIX_O", "R22", str(Tmatrix_o[1][1]))
    config.set("TMATRIX_O", "R23", str(Tmatrix_o[1][2]))

    config.set("TMATRIX_O", "R31", str(Tmatrix_o[2][0]))
    config.set("TMATRIX_O", "R32", str(Tmatrix_o[2][1]))
    config.set("TMATRIX_O", "R33", str(Tmatrix_o[2][2]))

    config.set("TMATRIX_O", "TX", str(Tmatrix_o[0][3]))
    config.set("TMATRIX_O", "TY", str(Tmatrix_o[1][3]))
    config.set("TMATRIX_O", "TZ", str(Tmatrix_o[2][3]))


    
    config.set("CAMERA_MATRIX", "FX", str(fx))
    config.set("CAMERA_MATRIX", "FY", str(fy))
    config.set("CAMERA_MATRIX", "CX", str(cx))
    config.set("CAMERA_MATRIX", "CY", str(cy))
    config.set("CAMERA_MATRIX", "DEPTH_SCALE", str(depth_scale))
    
    corner = corners[0].squeeze()
    
    config.set("ARUCO_CORNER", "TOPLEFT_X", str(int(corner[0][0])))
    config.set("ARUCO_CORNER", "TOPLEFT_Y", str(int(corner[0][1])))
    config.set("ARUCO_CORNER", "TOPRIGHT_X", str(int(corner[1][0])))
    config.set("ARUCO_CORNER", "TOPRIGHT_Y", str(int(corner[1][1])))
    config.set("ARUCO_CORNER", "BOTTOMRIGHT_X", str(int(corner[2][0])))
    config.set("ARUCO_CORNER", "BOTTOMRIGHT_Y", str(int(corner[2][1])))
    config.set("ARUCO_CORNER", "BOTTOMLEFT_X", str(int(corner[3][0])))
    config.set("ARUCO_CORNER", "BOTTOMLEFT_Y", str(int(corner[3][1])))
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    
    print("Calibration completed!")
