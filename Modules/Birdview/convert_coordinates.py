#!/usr/bin/env python3

import numpy as np
import yaml
import logging

# convert pixels in 2D image space into 3D world space
class ConvertCoordinates():
    def __init__(self, intrisic_params_file, calibration_aruco_file):
        # Read calibration files
        with open(intrisic_params_file, "r") as f:
            self.intrisic_params_   = yaml.load(f)

        with open(calibration_aruco_file, "r") as f:
            self.calib_aruco_       = yaml.load(f)

        # Realsense camera intrisic parameters
        self.fx_                    = self.intrisic_params_["Camera.fx"]
        self.fy_                    = self.intrisic_params_["Camera.fy"]
        self.cx_                    = self.intrisic_params_["Camera.cx"]
        self.cy_                    = self.intrisic_params_["Camera.cy"]

        # Transformation matrix wrt origin
        self.Tmatrix_o_             = self.calib_aruco_["Transformation_matrix_o"]
        self.Tmatrix_o_             = np.asarray(self.Tmatrix_o_, dtype=np.float)
        self.Tmatrix_o_             = np.reshape(self.Tmatrix_o_, (4, 4))
        print(self.Tmatrix_o_)

    def convert(self, pixel_point):
        # pixel in 2D image space
        point_u                     = (int)(pixel_point[0])
        point_v                     = (int)(pixel_point[1])
        point_depth                 = (float)(pixel_point[2])

        # 3D point w.r.t the camera
        coor_x_c                    = (point_u - self.cx_)/(self.fx_)*point_depth
        coor_y_c                    = (point_v - self.cy_)/(self.fy_)*point_depth
        coor_z_c                    = point_depth
        coor_c                      = np.array([coor_x_c, coor_y_c, coor_z_c, 1.0], dtype=np.float)
        coor_c                      = np.reshape(coor_c, (4, 1))

        # 3D point w.r.t the origin (Aruco marker)
        coor_o                      = np.matmul(self.Tmatrix_o_, coor_c)
        coor_o                      = np.reshape(coor_o, (4, 1))

        x_coor_o                    = coor_o[0][0]                     # x coordinate w.r.t the origin (Aruco marker)
        y_coor_o                    = coor_o[1][0]                     # y coordinate w.r.t the origin (Aruco marker)
        z_coor_o                    = coor_o[2][0]                     # z coordinate w.r.t the origin (Aruco marker)
        # point_3d                    = np.asarray((x_coor_o, y_coor_o, z_coor_o), dtype=np.float).reshape(1, 3)
        point_3d                    = (x_coor_o, y_coor_o, z_coor_o)

        # return
        return point_3d