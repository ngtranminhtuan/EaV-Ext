#!/usr/bin/env python3
import logging
# import rospy
import sys
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)

class RealsenseCamera:
    def __init__(self,
                 img_width=1280,
                 img_height=720,
                 fps=30):
        # Realsense camera config
        self.img_width_             = img_width
        self.img_height_            = img_height
        self.fps_ = fps

        self.pipeline_              = None
        self.config_                = None
        self.scale_                 = None
        self.cfg_                   = None
        self.rgb_profile_           = None
        self.intrinsic_paramas_     = None

        # Determine depth scale
        self.depth_scale_           = None
        self.color_sensor_cfg_      = None
        self.depth_sensor_cfg_      = None
        self.colorizer_             = None

        # Post-processing
        self.hole_filter_           = rs.hole_filling_filter()
        self.decimate_filter_       = rs.decimation_filter()
        self.pc_                    = rs.pointcloud()
        self.filters_               = [rs.disparity_transform(),
                                        rs.spatial_filter(),
                                        rs.temporal_filter(),
                                        rs.disparity_transform(False)]


    def connect(self):
        # Configure depth and color streams
        self.pipeline_              = rs.pipeline()
        self.config_                = rs.config()

        # self.config_.enable_device(str(self.device_id_))
        self.config_.enable_stream(rs.stream.depth, self.img_width_, self.img_height_, \
                                    rs.format.z16, self.fps_)         # depth image
        self.config_.enable_stream(rs.stream.color, self.img_width_, self.img_height_, \
                                    rs.format.rgb8, self.fps_)        # color image

        self.cfg_                   = self.pipeline_.start(self.config_)
        self.color_sensor_cfg_      = self.cfg_.get_device().query_sensors()[1]
        self.color_sensor_cfg_.set_option(rs.option.enable_auto_exposure, True)
        self.depth_sensor_cfg_      = self.cfg_.get_device().query_sensors()[0]
        self.depth_sensor_cfg_.set_option(rs.option.enable_auto_exposure, True)

        # Determine intrinsic parameters
        self.rgb_profile_           = self.cfg_.get_stream(rs.stream.color)
        self.intrinsic_paramas_     = self.rgb_profile_.as_video_stream_profile().get_intrinsics()
        print("self.intrinsic_paramas_ ====", self.intrinsic_paramas_)

        # Determine depth scale
        self.depth_scale_           = self.cfg_.get_device().first_depth_sensor().get_depth_scale()
        print("self.depth_scale_  ====", self.depth_scale_ )


        # Set min max value
        self.colorizer_ = rs.colorizer()
        self.colorizer_.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        self.colorizer_.set_option(rs.option.min_distance, 0.5)
        self.colorizer_.set_option(rs.option.max_distance, 2.5)

        # Post process
        self.hole_filter_           = rs.hole_filling_filter()
        self.decimate_filter_       = rs.decimation_filter()
        self.decimate_filter_.set_option(rs.option.filter_magnitude, 1)
        self.pc_                    = rs.pointcloud()
        self.filters_               = [rs.disparity_transform(),
                                        rs.spatial_filter(),
                                        rs.temporal_filter(),
                                        rs.disparity_transform(False)]

    def get_images_bundle(self):
        frames = self.pipeline_.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_frame = self.hole_filter_.process(aligned_depth_frame)
        aligned_depth_frame = self.decimate_filter_.process(aligned_depth_frame)
        for f in self.filters_:
            aligned_depth_frame = f.process(aligned_depth_frame)

        # RGB color image
        color_image = np.asanyarray(aligned_color_frame.get_data(), dtype=np.uint8)

        # depth image
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.uint16)
        depth_image = np.expand_dims(depth_image, axis=2)
        depth_image_color = np.asanyarray(self.colorizer_.colorize(aligned_depth_frame).get_data())

        return {
            'color_image': color_image,
            'depth_image': depth_image,
            'depth_image_color': depth_image_color,
        }


    def get_depth_scale(self):
        return self.depth_scale_