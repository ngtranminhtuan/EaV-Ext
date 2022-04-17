import numpy as np
import pyrealsense2 as rs
import sys
import os
import argparse
import cv2
import threading
import queue
import time
import glob

from Utils.Communication.app import *
from Utils.Performance_Measurement.app import Logging
from Utils.Config.app import getConfig
from Utils.Cv2_Effect.app import *

from Modules.People_Counting.app import *
from Modules.Birdview.compute_birdview import *
sys.path.append("Modules/People_Detection")
from Modules.People_Detection.app import drawAnimationCircle
sys.path.append("Modules/Face_Detection")
from Modules.Face_Detection.app import mergeFaces, locateFaces, replaceAnimeFaces
sys.path.append("Modules/Action_Detection")
from Modules.Action_Detection.app import runActionsFilter
from Modules.Action_Detection.action_detector import drawPose, drawAction, EDGES

def initCamera():
    # Configure depth and color streams
    pipeline    = rs.pipeline()
    config      = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper    = rs.pipeline_wrapper(pipeline)
    pipeline_profile    = config.resolve(pipeline_wrapper)
    device              = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    imgProfile = pipeline.start(config)
    return pipeline, imgProfile

def readCamera(camMode, cap, imgQueue):
    preFrameNum = -1
    while True:
        orgImg = None
        if camMode:
            ret, frames = cap.try_wait_for_frames()
            # Skip old frame by previous frame number
            if preFrameNum == frames.get_color_frame().get_frame_number():
                continue
            preFrameNum   = frames.get_color_frame().get_frame_number()
            alignedFrames  = alignToColor.process(frames)
            colorFrame     = alignedFrames.get_color_frame()

            aligned_depth_frame = alignedFrames.get_depth_frame()
            depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.uint16)

            if not colorFrame:
                continue
            orgImg = np.asanyarray(colorFrame.get_data())
        else:
            ret, orgImg  = cap.read()
            time.sleep(0.2)
        # Put image into queue
        if orgImg is not None:
            imgQueue.put([orgImg, depth_image])
        else:
            imgQueue.put(None)
        # Delete older images
        if imgQueue.qsize() > 1:
            _ = imgQueue.get()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',default=True, action='store_true', help='Active debug mode, use camera 0 as input')
    parser.add_argument('--video',default="", help='Enable test video')
    args = parser.parse_args()

    debugMode = args.debug
    videoPath = args.video

     # Init logging to measure performance
    logging = Logging(debug=debugMode)

    # Get config vars
    config = getConfig()
    streamPort    = int(config['COMMUNICATION']['STREAM_PORT'])
    pubPeoplePort = int(config['COMMUNICATION']['PUB_PEOPLE_IMG_PORT'])
    subPeoplePort = int(config['COMMUNICATION']['SUB_PEOPLE_RESULT_PORT'])
    pubFacePort   = int(config['COMMUNICATION']['PUB_FACE_IMG_PORT'])
    subFacePort   = int(config['COMMUNICATION']['SUB_FACE_RESULT_PORT'])
    pubActionPort   = int(config['COMMUNICATION']['PUB_ACTION_IMG_PORT'])
    subActionPort   = int(config['COMMUNICATION']['SUB_ACTION_RESULT_PORT'])

    # Init sockets
    streamPublisher  = initPublisher(streamPort)
    peoplePublisher  = initPublisher(pubPeoplePort)
    peopleSubscriber = initSubsciber(subPeoplePort, "PEOPLE_RESULT")
    facePublisher    = initPublisher(pubFacePort)
    faceSubscriber   = initSubsciber(subFacePort, "FACE_RESULT")
    actionPublisher  = initPublisher(pubActionPort)
    actionSubscriber = initSubsciber(subActionPort, "ACTION_RESULT")

    # Get other configs
    showOutput           = True if config['DEFAULT']['SHOW_OUTPUT']=="1" else False
    camMode              = True if config['DEFAULT']['CAM_MODE']=="1" else False
    if videoPath == "":
        videoPath        = config['DEFAULT']['VIDEO_PATH']
    fps                  = int(config['DEFAULT']['FPS'])
    outputDir            = config['DEFAULT']['OUTPUT_DIR']
    selectCountingZone   = True if config['DEFAULT']['SELECT_GATE_POSITION']=="1" else False
    gatePosition         = config['DEFAULT']['GATE_POSITION'].split(",")
    gatePosition         = list(map(int, gatePosition))
    doorPosition         = config['DEFAULT']['DOOR_POSITION'].upper()
    numFilter            = int(config['DEFAULT']['NUM_FILTER'])
    
    if not (doorPosition=="TOP" or doorPosition=="BOTTOM"):
        print("Door position not support, please check config.ini file!")
        exit()    

    # For batch testing
    # videoPath = args.video
    if camMode:
        cap, imgProfile = initCamera()
        alignToColor  = rs.align(align_to=rs.stream.color)
        for i in range(10):
            ret, frames = cap.try_wait_for_frames()
        alignedFrames = alignToColor.process(frames)
        colorFrame    = alignedFrames.get_color_frame()
        orgImg        = np.asanyarray(colorFrame.get_data())

        # Depth frame process
        # Preprocess

        # hole_filter_           = rs.hole_filling_filter()
        # decimate_filter_       = rs.decimation_filter()
        # decimate_filter_.set_option(rs.option.filter_magnitude, 1)
        # filters_               = [rs.disparity_transform(),
        #                                 rs.spatial_filter(),
        #                                 rs.temporal_filter(),
        #                                 rs.disparity_transform(False)]

        aligned_depth_frame = alignedFrames.get_depth_frame()
        # aligned_depth_frame = hole_filter_.process(aligned_depth_frame)
        # aligned_depth_frame = decimate_filter_.process(aligned_depth_frame)
        # for f in filters_:
        #     aligned_depth_frame = f.process(aligned_depth_frame)
        
        # Depth image
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.uint16)
        depth_scale = imgProfile.get_device().first_depth_sensor().get_depth_scale()
    else:
        cap = cv2.VideoCapture(videoPath)
        for i in range(10):
            ret, orgImg = cap.read()
    h,w,_ = orgImg.shape

    if selectCountingZone:
        selectZone(config, orgImg)
    
    # Create video writer
    if outputDir:
        if camMode:
            outputPath = os.path.join(outputDir, "output_cam.avi")
        else:
            videoName = os.path.splitext(videoPath)[0]
            videoName = os.path.basename(videoName)
            outputPath = os.path.join(outputDir, videoName + ".avi")
        # print(outputPath)
        # print(w, h, fps)
        # exit()
        res_video = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w,h))

    imgW, imgH   = orgImg.shape[1], orgImg.shape[0]
    topZone      = [0, 0, imgW, gatePosition[1]]
    botZone      = [0, gatePosition[1], imgW, imgH]
    preTopList   = []
    preBotList   = []
    upCounter    = 0
    downCounter  = 0
    historyDict  = {}
    frameIdx     = -1
    previousActions = {}
    
    # Init birdview
    birdview = BirdView()

    # Init thread read camera
    imgQueue = queue.Queue()

    # Run thread
    threading.Thread(target=readCamera, args=(camMode, cap, imgQueue), daemon=True).start()

    # Load anime faces
    animeFacePaths = glob.glob(os.path.join("Resources", "10_Anime_Faces", "*.jpg"))
    animeImgs = []
    for animeFacePath in animeFacePaths:
        img = cv2.imread(animeFacePath)
        animeImgs.append(img)
    
    # Load bounding boxes animation images
    dictStates = {}
    animationImgPaths = glob.glob(os.path.join("Resources", "Bounding_Boxes_Animation_Images", "*.png"))
    animationImgs = []
    for animationImgPath in animationImgPaths:
        img = cv2.imread(animationImgPath, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        animationImgs.append(img)

    print("Application starting...")
    while True:
        logging.start("Total")

        logging.start("Read")
        orgImg, depth_image = imgQueue.get()
        if orgImg is None:
            break
        frameIdx += 1        
        logging.end("Read")

        logging.start("Write")
        res_video.write(orgImg)
        np.save(os.path.join(outputDir,"Depth", str(frameIdx)+".npy"), depth_image)
        logging.end("Write")

        cv2.imshow("Result", orgImg)
        k = cv2.waitKey(1)
        if k == 27:
            break
        

        logging.end("Total")
        logging.print_result()
        continue

        logging.start("Send P")
        import time
        pubSendArray(peoplePublisher, "", orgImg)
        logging.end("Send P")

        logging.start("Get P")
        outputTrackingBboxes = subRecvArray(peopleSubscriber)
        logging.end("Get P")

        logging.start("Couting")
        result, history = countPeople(historyDict, preTopList, preBotList, upCounter, downCounter, numFilter, outputTrackingBboxes, doorPosition, imgW, imgH, topZone, botZone)
        numIn, upCounter, downCounter = result
        historyDict, preTopList, preBotList = history
        logging.end("Couting")

        logging.start("Merge Faces")
        mergedImg, outputPeopleBoxes, peopleImgs = mergeFaces(outputTrackingBboxes, orgImg)
        logging.end("Merge Faces")

        logging.start("Send F")
        pubSendArray(facePublisher, "", mergedImg)
        logging.end("Send F")

        logging.start("Get F")
        outputFaceBboxes = subRecvArray(faceSubscriber)
        logging.end("Get F")

        logging.start("Locate F")
        newFaceBoxes, orgImg = locateFaces(outputFaceBboxes, orgImg, outputPeopleBoxes, peopleImgs)
        logging.end("Locate F")

        logging.start("Anime F")
        orgImg = replaceAnimeFaces(orgImg, newFaceBoxes, animeImgs)
        logging.end("Anime F")

        logging.start("Get A")
        results = recvZippedPickle(actionSubscriber)
        poseKeypoints      = results["poseKeypoints"]
        actionResults      = results["actionResults"]
        skeletons          = results["skeletons"]
        middleBottomPoints = results["middleBottomPoints"]
        nosePoints = results["nosePoints"]
        logging.end("Get A")

        logging.start("Filter Noise A")
        actionResults, previousActions = runActionsFilter(actionResults, previousActions)
        logging.end("Filter Noise A")

        logging.start("Draw A")
        drawPose(orgImg, poseKeypoints, EDGES, confidence_threshold=0.3)
        orgImg = drawAction(orgImg, actionResults, skeletons)
        logging.end("Draw A")
    
        logging.start("Draw")
        # Draw people counting informations
        cv2.putText(orgImg, "Up: "+str(upCounter),(10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 2)
        cv2.putText(orgImg, "Down: "+str(downCounter),(10, 100), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 2)
        cv2.putText(orgImg, "Num In: "+str(numIn),(10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,255,0), 2)
        cv2.line(orgImg, (0, gatePosition[1]), (imgW, gatePosition[1]), (51,255,255), 2)
        # Draw people bouding boxes and animation circle
        dictStates, orgImg, filteredOutputTrackingBboxes = drawAnimationCircle(orgImg, middleBottomPoints, dictStates, animationImgs, outputTrackingBboxes)
        logging.end("Draw")

        logging.start("Tranform BirdView")
        # birdview.show_people_on_topview(orgImg, birdview_image, outputTrackingBboxes)
        # birdview_image = birdview.show_topview(filteredOutputTrackingBboxes)
        birdview_image = birdview.convert_centroid2birdview(orgImg, outputTrackingBboxes, depth_image, depth_scale)

        # Draw bb person
        # for person in outputTrackingBboxes:
        #     x0, y0, x1, y1 = int(person[0]),int(person[1]),int(person[2]),int(person[3])
        #     # Ignore invalid bounding boxes
        #     if (y1-y0 < 1) or (x1-x0 < 1) or (x0<0) or (y0<0) or (x1>=imgW) or (y1>=imgH):
        #         continue
        #     orgImg = cv2.rectangle(orgImg, (x0,y0), (x1,y1), (0,0,255), 3)
        logging.end("Tranform BirdView")

        # Display result
        if showOutput:
            logging.start("Display")
            cv2.namedWindow("Birdview", cv2.WINDOW_NORMAL)
            cv2.imshow('Birdview', birdview_image)
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.imshow('Output',orgImg)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break
            elif k == 32:
                print("Paused!")
                while True:
                    k2 = cv2.waitKey(1)
                    if k2 == 32:
                        break
            logging.end("Display")

        # Send to streaming server
        logging.start("Send S")
        orgImg=np.ascontiguousarray(orgImg)
        orgImg = np.concatenate((orgImg, birdview_image), axis=1)
        pubSendArray(streamPublisher, "", orgImg)
        logging.end("Send S")
        logging.end("Total")

        # Write video
        if outputDir:
            logging.start("Write video")
            res_video.write(orgImg)
            logging.end("Write video")

        # Print mean time table
        # logging.print_result()
        logging.print_mean_result()

    # Release resources
    if outputDir:
        res_video.release()
