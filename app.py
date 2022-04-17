from pyexpat.errors import XML_ERROR_UNDECLARING_PREFIX
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
from Modules.People_Detection.app import drawAnimationCircle, trackPeoples
sys.path.append("Modules/Face_Detection")
from Modules.Face_Detection.app import mergeFaces, locateFaces, replaceAnimeFaces, trackFaces
sys.path.append("Modules/Action_Detection")
from Modules.Action_Detection.app import runActionsFilter
from Modules.Action_Detection.action_detector import drawPose, drawAction, EDGES
sys.path.append("Modules/StyleTransfer")
from Modules.StyleTransfer.app import *
from Algorithms.Tracking.sort import Sort
sys.path.append("Modules/People_Counting")
from Modules.People_Counting.app import addIcon

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

def readCamera(camMode, cap, imgQueue, cropMode, videoPath, playAdsVideo):
    preFrameNum = -1

    frameIdx = 10
    videoDir = os.path.dirname(videoPath)
    depthDir = os.path.join(videoDir, "Depth")
    while True:
        if playAdsVideo:
            continue

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
            ret, orgImg = cap.read()
            depthPath = os.path.join(depthDir, str(frameIdx)+".npy")
            if os.path.exists(depthPath):
                depth_image = np.load(depthPath)
            else:
                depth_image = np.zeros(shape=(720,1280))
            frameIdx += 1
            time.sleep(0.15)
        # Put image into queue
        if orgImg is not None:
            if cropMode:
                xCenter = orgImg.shape[1] // 2
                yCenter = orgImg.shape[0] // 2
                startX = xCenter-orgImg.shape[1]//4
                endX   = xCenter+orgImg.shape[1]//4
                startY = yCenter-orgImg.shape[0]//4
                endY   = yCenter+orgImg.shape[0]//4
                cropImg = orgImg[startY:endY, startX:endX, :]
                cropImg =np.ascontiguousarray(cropImg)
                # cropDepth = depth_image[startY:endY, startX:endX]
                cropImg = cv2.resize(cropImg, (1280,720), interpolation=cv2.INTER_LINEAR)
                # cropDepth = cv2.resize(cropDepth, (1280,720), interpolation=cv2.INTER_NEAREST)
                imgQueue.put([orgImg, depth_image, cropImg])
            else:
                # Normal mode
                imgQueue.put([orgImg, depth_image])
        else:
            if cropMode:
                imgQueue.put([None, None, None])
            else:
                imgQueue.put([None, None])
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
    pubStylePort    = int(config['COMMUNICATION']['PUB_STYLE_IMG_PORT'])
    subStylePort   = int(config['COMMUNICATION']['SUB_STYLE_RESULT_PORT'])

    # Birdview Mode enable/disable
    bvMode         = int(config['BIRDVIEW']['BV_MODE'])

    # Init sockets
    streamPublisher  = initPublisher(streamPort)
    peoplePublisher  = initPublisher(pubPeoplePort)
    peopleSubscriber = initSubsciber(subPeoplePort, "PEOPLE_RESULT")
    facePublisher    = initPublisher(pubFacePort)
    faceSubscriber   = initSubsciber(subFacePort, "FACE_RESULT")
    actionPublisher  = initPublisher(pubActionPort)
    actionSubscriber = initSubsciber(subActionPort, "ACTION_RESULT")
    stylePublisher   = initPublisher(pubStylePort)
    styleSubscriber = initSubsciber(subStylePort, "STYLE_RESULT")

    # Get other configs
    showOutput           = True if config['DEFAULT']['SHOW_OUTPUT']=="1" else False
    camMode              = True if config['DEFAULT']['CAM_MODE']=="1" else False
    videoPath            = config['DEFAULT']['VIDEO_PATH'] if args.video=="" else args.video
    fps                  = int(config['DEFAULT']['FPS'])
    outputDir            = config['DEFAULT']['OUTPUT_DIR']
    selectCountingZone   = True if config['DEFAULT']['SELECT_GATE_POSITION']=="1" else False
    gatePosition         = config['DEFAULT']['GATE_POSITION'].split(",")
    gatePosition         = list(map(int, gatePosition))
    doorPosition         = config['DEFAULT']['DOOR_POSITION'].upper()
    numFilter            = int(config['DEFAULT']['NUM_FILTER'])
    cropMode             = True if config['DEFAULT']['CROP_MODE']=="1" else False
    
    # Icon size
    iconSize             = int(config['DEFAULT']['ICON_SIZE'])
    textSize             = float(config['DEFAULT']['TEXT_SIZE'])


    # Load In, Out and Number of human Icon Image
    upIcon      = cv2.imread("Resources/In_Out_Icon/Up.png", -1)
    downIcon    = cv2.imread("Resources/In_Out_Icon/Down.png", -1)
    numInIcon   = cv2.imread("Resources/In_Out_Icon/NumIn.png", -1)

    upIcon = cv2.resize(upIcon, (iconSize, iconSize))
    downIcon = cv2.resize(downIcon, (iconSize, iconSize))
    numInIcon = cv2.resize(numInIcon, (iconSize, iconSize))

    # GAN time and Normal time
    ganTime              = int(config['DEFAULT']['GAN_TIME'])
    normalTime           = int(config['DEFAULT']['NORMAL_TIME'])
    faceMode             = True if config['FACE_DETECTION']['FACE_MODE']=="1" else False
    
    if not (doorPosition=="TOP" or doorPosition=="BOTTOM"):
        print("Door position not support, please check config.ini file!")
        exit()    

    # For batch testing
    if camMode:
        cap, imgProfile = initCamera()
        alignToColor  = rs.align(align_to=rs.stream.color)
        for i in range(10):
            ret, frames = cap.try_wait_for_frames()
        alignedFrames = alignToColor.process(frames)
        colorFrame    = alignedFrames.get_color_frame()
        orgImg        = np.asanyarray(colorFrame.get_data())
        aligned_depth_frame = alignedFrames.get_depth_frame()
        
        # Depth image
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.uint16)
        depth_scale = imgProfile.get_device().first_depth_sensor().get_depth_scale()
    else:
        cap = cv2.VideoCapture(videoPath)
        for i in range(10):
            ret, orgImg = cap.read()
        depth_image = np.zeros(shape=(720,1280))
        depth_scale = 0.001
    h,w,_ = orgImg.shape

    # Crop image mode
    if cropMode:
        xCenter = orgImg.shape[1] // 2
        yCenter = orgImg.shape[0] // 2
        startX = xCenter-orgImg.shape[1]//4
        endX   = xCenter+orgImg.shape[1]//4
        startY = yCenter-orgImg.shape[0]//4
        endY   = yCenter+orgImg.shape[0]//4
        cropImg = orgImg[startY:endY, startX:endX, :]
        cropImg =np.ascontiguousarray(cropImg)
        orgImg = cv2.resize(cropImg, (1280,720), interpolation=cv2.INTER_LINEAR)

    if selectCountingZone:
        selectZone(config, orgImg)
    
    # Create video writer
    if outputDir:
        if camMode:
            outputPath = os.path.join(outputDir, "output_cam.mp4")
        else:
            videoName = os.path.splitext(videoPath)[0]
            videoName = os.path.basename(videoName)
            outputPath = os.path.join(outputDir, videoName + "_out.avi")
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
    playAdsVideo = 0
    threading.Thread(target=readCamera, args=(camMode, cap, imgQueue, cropMode, videoPath, playAdsVideo), daemon=True).start()

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

    # Load dmp logo
    dmpLogo = cv2.imread("Resources/DMP_Logo/dmp_logo.png")
    dmpLogo = cv2.resize(dmpLogo, (235, 100))

    # Create trackers
    peopleTracker = Sort(max_age=1)
    faceTracker   = Sort(max_age=1)

    ganMode = True
    ganModeStart = 0
    normalModeStart = 0

    birdview_image = np.full([imgH, imgW, 3], 255, dtype=np.uint8)

    # Play ads videos
    adsCapture = None
    orgImg = None
    startWaving1 = 0
    startWaving2 = 0
    print("Application starting...")
    while True:
        logging.start("Total")
        if playAdsVideo:
            videoPath = ""
            if adsCapture is None:
                # Play 6 seconds video
                if playAdsVideo == 1:
                    videoPath = "Resources/Ads_Videos/6s_video.mp4"
                # Play 20 seconds video
                elif playAdsVideo == 2:
                    videoPath = "Resources/Ads_Videos/20s_video.mp4"
                adsCapture = cv2.VideoCapture(videoPath)
                
            s = time.time()
            ret, orgImg = adsCapture.read()
            if not ret:
                playAdsVideo = 0
                adsCapture = None
                continue
            e = time.time() - s
            if e < 0.0417:
                time.sleep(0.0417-e)
        else:
            logging.start("Read")
            if cropMode:
                orgImg, depth_image, cropImg = imgQueue.get()
                if orgImg is None:
                    break
            else:
                orgImg, depth_image  = imgQueue.get()
                if orgImg is None:
                    break
            frameIdx += 1        
            logging.end("Read")
            
            if ganMode:
                # Switch to GAN mode
                if ganModeStart == 0:
                    ganModeStart = time.time()
                ganModeEnd = time.time()
                if (ganModeEnd - ganModeStart > ganTime):
                    ganMode = False
                    ganModeStart = 0
                    continue

                logging.start("Send GAN")
                pubSendArray(stylePublisher, "", orgImg)
                logging.end("Send GAN")

                logging.start("Get GAN")
                orgImg = subRecvArray(styleSubscriber)
                logging.end("Get GAN")
            else:
                # Switch to nomal mode
                if normalModeStart == 0:
                    normalModeStart = time.time()
                normalModeEnd = time.time()
                if (normalModeEnd - normalModeStart > normalTime):
                    ganMode = True
                    normalModeStart = 0
                    continue

                logging.start("Send P+A")
                if cropMode:
                    pubSendArray(actionPublisher, "", orgImg)
                    pubSendArray(peoplePublisher, "", cropImg)
                else:
                    pubSendArray(actionPublisher, "", orgImg)
                    pubSendArray(peoplePublisher, "", orgImg)
                logging.end("Send P+A")

                logging.start("Get P")
                outputTrackingBboxes = subRecvArray(peopleSubscriber)
                logging.end("Get P")

                logging.start("Get A")
                results = recvZippedPickle(actionSubscriber)
                poseKeypoints      = results["poseKeypoints"]
                actionResults      = results["actionResults"]
                skeletons          = results["skeletons"]
                middleBottomPoints = results["middleBottomPoints"]
                nosePoints = results["nosePoints"]
                logging.end("Get A")

                # Use skeleton for people detection
                peopleBoxes = []
                for skeleton in poseKeypoints:

                    minx = 999
                    miny = 999
                    maxx = -999
                    maxy = -999

                    for keypoint in skeleton:
                        y, x, score = keypoint
                        minx = min(minx, x)
                        maxx = max(maxx, x)
                        miny = min(miny, y)
                        maxy = max(maxy, y)

                    x0 = int(minx * 1280)
                    y0 = int(miny * 720)
                    x1 = int(maxx * 1280)
                    y1 = int(maxy * 720)

                    # Padding box
                    x0 = x0 -  int(0.075 * abs(x1-x0))
                    y0 = y0 -  int(0.075 * abs(y1-y0))
                    x1 = x1 +  int(0.075 * abs(x1-x0))
                    y1 = y1 +  int(0.075 * abs(y1-y0))

                    avgScore = np.average(np.array(skeleton), axis=0)[2]
                    peopleBoxes.append([0, avgScore, x0,y0,x1,y1])
                # Track people
                outputTrackingBboxes = trackPeoples(peopleTracker, peopleBoxes)

                if cropMode:
                    logging.start("Transform P")
                    newOutputTrackingBboxes = []
                    for outputTrackingBbox in outputTrackingBboxes:
                        x0, y0, x1, y1, id = outputTrackingBbox
                        newX0, newX1 = (x0 // 2) + 320, (x1 // 2) + 320
                        newY0, newY1 = (y0 // 2) + 180, (y1 // 2) + 180
                        if newX0 >= 1280:
                            newX0 = 1279
                        if newY0 >= 1280:
                            newY0 = 1279
                        if newX1 >= 720:
                            newX1 = 719
                        if newY1 >= 720:
                            newY1 = 719
                        newOutputTrackingBboxes.append([newX0, newY0, newX1, newY1, id])
                    outputTrackingBboxes = newOutputTrackingBboxes
                    logging.end("Transform P")

                logging.start("Couting")
                result, history = countPeople(historyDict, preTopList, preBotList, upCounter, downCounter, numFilter, outputTrackingBboxes, doorPosition, imgW, imgH, topZone, botZone)
                numIn, upCounter, downCounter = result
                historyDict, preTopList, preBotList = history
                logging.end("Couting")

                logging.start("Merge F")
                mergedImg, outputPeopleBoxes, peopleImgs = mergeFaces(outputTrackingBboxes, orgImg)
                logging.end("Merge F")

                logging.start("Send F")
                pubSendArray(facePublisher, "", mergedImg)
                logging.end("Send F")

                logging.start("Get F")
                outputFaceBboxes = subRecvArray(faceSubscriber)
                logging.end("Get F")

                logging.start("Locate F")
                newFaceBoxes, orgImg = locateFaces(outputFaceBboxes, orgImg, outputPeopleBoxes, peopleImgs)
                logging.end("Locate F")

                # Use skeleton for face bounding boxes
                poseThreshold = 0.15
                poseFaceBoxes = []
                for skeleton in poseKeypoints:
                    noseY, noseX, noseScore = skeleton[0]
                    leftShoulderY, leftShoulderX, leftShoulderScore = skeleton[5]
                    if noseScore > poseThreshold and leftShoulderScore > poseThreshold:
                        boxRadius = (leftShoulderY - noseY) * 0.5
                        x0 = int((noseX-boxRadius)*imgW)
                        y0 = int((noseY-boxRadius)*imgH)
                        x1 = int((noseX+boxRadius)*imgW)
                        y1 = int((noseY+boxRadius)*imgH)
                        # Ignore invalid boxes
                        if x0 > x1 or y0 > y1:
                            continue
                        avgScore = np.average(np.array(skeleton), axis=0)[2]
                        faceBox = [0, avgScore, x0, y0, x1, y1]
                        poseFaceBoxes.append(faceBox)
                newFaceBoxes = trackFaces(faceTracker, poseFaceBoxes).tolist()

                logging.start("Filter A")
                actionResults, previousActions = runActionsFilter(actionResults, previousActions)
                logging.end("Filter A")

                WAVING_DURATION = 3
                FORGET_DURATION = 5
                numPeopleWaving = 0
                # Count number of people waving hand
                for id, action in actionResults.items():
                    if action=="wave":
                        numPeopleWaving += 1
                # Forget if long time waving
                if numPeopleWaving >= 3 and time.time()-startWaving2 > FORGET_DURATION:
                    startWaving2 = time.time()
                elif 0 < numPeopleWaving < 3 and time.time()-startWaving1 > FORGET_DURATION:
                    startWaving1 = time.time()

                # Detetermine waving is enough to play ads video
                if numPeopleWaving >= 3 and time.time()-startWaving2 > WAVING_DURATION:
                    if startWaving2 == 0:
                        startWaving2 = time.time()
                    playAdsVideo = 2
                    startWaving2 = 0
                    continue
                elif 0 < numPeopleWaving < 3 and time.time()-startWaving1 > WAVING_DURATION:
                    if startWaving1 == 0:
                        startWaving1 = time.time()
                    playAdsVideo = 1
                    startWaving1 = 0
                    continue
                
                logging.start("Draw")
                # Draw skeleton keypoints
                drawPose(orgImg, poseKeypoints, EDGES, confidence_threshold=0.3)
                # Draw action results
                orgImg = drawAction(orgImg, actionResults, skeletons)
                
                # Draw Up, Down, NumIn Icon
                orgImg = addIcon(orgImg, upIcon, downIcon, numInIcon, upCounter, downCounter, numIn, textSize=textSize, iconSize=(iconSize, iconSize))

                cv2.line(orgImg, (0, gatePosition[1]), (imgW, gatePosition[1]), (51,255,255), 2)
                # Draw people bouding boxes and animation circle
                dictStates, orgImg = drawAnimationCircle(orgImg, middleBottomPoints, dictStates, animationImgs, outputTrackingBboxes)

                # Replace animate face
                if faceMode:
                    orgImg = replaceAnimeFaces(orgImg, newFaceBoxes, animeImgs)
                logging.end("Draw")
                
                if bvMode == 1:
                    logging.start("BV")
                    birdview_image = birdview.convert_centroid2birdview(orgImg, outputTrackingBboxes, depth_image, depth_scale)
                    logging.end("BV")

            # Insert DMP Logo
            orgImg[orgImg.shape[0]-dmpLogo.shape[0]:, orgImg.shape[1]-dmpLogo.shape[1]:,:] = dmpLogo

        # Display result
        if showOutput:
            logging.start("Display")

            if bvMode == 1:
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

        # Add text no record data
        cv2.putText(orgImg, "No data or videos are recorded",(int(imgW * 0.015), int(0.98*imgH)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Send to streaming server
        logging.start("Send S")
        orgImg=np.ascontiguousarray(orgImg)
        if not playAdsVideo:
            if cropMode:
                # Compute crop region
                xCenter = orgImg.shape[1] // 2
                yCenter = orgImg.shape[0] // 2
                startX = xCenter-orgImg.shape[1]//4
                endX   = xCenter+orgImg.shape[1]//4
                startY = yCenter-orgImg.shape[0]//4
                endY   = yCenter+orgImg.shape[0]//4
                # Crop
                orgImg = orgImg[startY:endY, startX:endX, :]
                birdview_image = birdview_image[startY:endY, startX:endX, :]
                # Compute contiguou
                orgImg =np.ascontiguousarray(orgImg)
                birdview_image =np.ascontiguousarray(birdview_image)
                # Resize
                orgImg = cv2.resize(orgImg, (1280,720), interpolation=cv2.INTER_LINEAR)
                birdview_image = cv2.resize(birdview_image, (1280,720), interpolation=cv2.INTER_LINEAR)
                
            if bvMode == 1 and not ganMode:
                # Resize birdview_image
                birdview_image = cv2.resize(birdview_image, (427,240))
                orgImg[:birdview_image.shape[0],orgImg.shape[1]-birdview_image.shape[1]:,:] = birdview_image

        pubSendArray(streamPublisher, "", orgImg)
        logging.end("Send S")
        logging.end("Total")

        # Write video
        if outputDir:
            logging.start("Write")
            res_video.write(orgImg)
            logging.end("Write")

        # Print mean time table
        # logging.print_result()
        logging.print_mean_result()



    # Release resources
    if outputDir:
        res_video.release()
