import sys
sys.path.append("../..")

import numpy as np
import argparse


from Utils.Communication.app import *
from Utils.Performance_Measurement.app import Logging
from Utils.Config.app import getConfig
from Utils.Cv2_Effect.app import *
from Utils.Gpu.app import splitGPU, initFP16, initXLA
from Algorithms.Tracking.sort import Sort


from people_detector import PeopleDetector

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isFP16)


def euclideanDistance(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)

def trackPeoples(tracker, bboxes, isDisabled=False):
    trackingBboxes = []
    for bbox in bboxes:
        # className   = bbox[0]
        score       = bbox[1]
        xMin, yMin  = bbox[2], bbox[3]
        xMax, yMax  = bbox[4], bbox[5]
        trackingBboxes.append([xMin, yMin, xMax, yMax, score])
    if isDisabled:
        outputTrackingBboxes = trackingBboxes
    else:
        if len(trackingBboxes) > 0:
            outputTrackingBboxes = tracker.update(np.array(trackingBboxes))
        else:
            outputTrackingBboxes = tracker.update()
    outputTrackingBboxes = np.round(outputTrackingBboxes)
    outputTrackingBboxes = outputTrackingBboxes.astype(int)

    return outputTrackingBboxes

def drawAnimationCircle(orgImg, middleBottomPoints, dictStates, animationImgs, outputTrackingBboxes):

    # Convert BGR to RGBA to concanate image and animation circle image
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGBA)

    # Loop all middle legs points
    for middleBottomPoint in middleBottomPoints:
      
        x0, x1 = middleBottomPoint[2], middleBottomPoint[3]
        y0, y1 = middleBottomPoint[4], middleBottomPoint[5]

        # orgImg = cv2.circle(orgImg, (x0,y0), 10, (255,0,0), -1)
        # orgImg = cv2.circle(orgImg, (x1,y1), 10, (255,0,0), -1)

        # Get current middle legs point
        legMidX = middleBottomPoint[0]
        legMidY = middleBottomPoint[1]

        # Find id with current index
        crrId = 1
        for outputTrackingBbox in outputTrackingBboxes:
            x0People, y0People, x1People, y1People, id = outputTrackingBbox
            if x0People < legMidX < x1People and y0People < legMidY < y1People:
                crrId = id
                break
        
        if str(crrId) not in dictStates:
            continue
        
        # Check dictStates contains states of circle animation
        if str(crrId) not in dictStates:
            dictStates[str(crrId)] = 0
        else:
            dictStates[str(crrId)] +=1
            if dictStates[str(crrId)] == 3:
                dictStates[str(crrId)] = 0

        # Get current animation
        animationImg = animationImgs[dictStates[str(id)]]

        x0Temp = middleBottomPoint[2]
        y0Temp = legMidY - (middleBottomPoint[3]-middleBottomPoint[2])//4
        x1Temp = legMidX + (middleBottomPoint[3]-middleBottomPoint[2])//2
        y1Temp = legMidY + (middleBottomPoint[3]-middleBottomPoint[2])//4

        # rectangle(orgImg, (x0Temp,y0Temp),(x1Temp,y1Temp), (246, 190, 0), 3)
        point = ((x0Temp+y0Temp)//2, (y0Temp+y1Temp)//2)
        isValidPoint = False
        for outputTrackingBbox in outputTrackingBboxes:
            x0, y0, x1, y1, id = outputTrackingBbox
            if x0 < point[0] < x1 and y0 < point[1] < y1:
                isValidPoint = True
        
        if not isValidPoint:
            continue

        # print(x0Temp, y0Temp, x1Temp, y1Temp)
        
        # # Check valid region
        if y0Temp < 0 or x1Temp>=orgImg.shape[1] or y1Temp>=orgImg.shape[0] or x1Temp<=x0Temp or y1Temp<=y0Temp:
            continue
        animationImg = cv2.resize(animationImg, (x1Temp-x0Temp, y1Temp-y0Temp))
        alpha_background = orgImg[y0Temp:y1Temp,x0Temp:x1Temp,3] / 255.0
        alpha_foreground = animationImg[:,:,3] / 255.0

        for color in range(0, 3):
            orgImg[y0Temp:y1Temp,x0Temp:x1Temp,color] = alpha_foreground  * animationImg[:,:,color]  + alpha_background * orgImg[y0Temp:y1Temp,x0Temp:x1Temp,color] * (1 - alpha_foreground)

    # Draw bounding boxes
    for outputTrackingBbox in outputTrackingBboxes:
        x0, y0, x1, y1, id = outputTrackingBbox
        rectangle(orgImg, (x0,y0),(x1,y1), (246, 190, 0), 3)

    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_RGBA2BGR)
    return dictStates, orgImg

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',default=False ,action='store_true', help='Active debug mode, use camera 0 as input')
    args = parser.parse_args()
    debugMode = args.debug

    # Init logging to measure performance
    logging = Logging(debugMode)

    # Get config yolo
    yoloVersion = config["DEFAULT"]["VERSION"]
    labelsPath  = config["DEFAULT"]["LABELS_PATH"]
    scoreThresh = float(config["DEFAULT"]["SCORE_THRESH"])
    nmsThresh   = float(config["DEFAULT"]["NMS_THRESH"])

    # Split GPU
    vRamUsage   = int(config["PERFORMANCE"]["VRAM"])
    splitGPU(vRamUsage)

    # Create tracking object
    keepFrame = int(config['TRACKING']['KEEP_FRAME'])
    tracker   = Sort(max_age=keepFrame)
    
    # Init pusher to push img to features
    subImgPort = int(config["COMMUNICATION"]["SUB_IMG_PORT"])
    imgSubsciber = initSubsciber(subImgPort, "ORG_IMG")
    pubResultPort = int(config["COMMUNICATION"]["PUB_RESULT_PORT"])
    resultPublisher = initPublisher(pubResultPort)

    # Init model
    if yoloVersion == "FULL":
        weightPath = config["FULL_VERSION"]["WEIGHT_PATH"]
        modelW = int(config["FULL_VERSION"]["MODEL_W"])
        modelH = int(config["FULL_VERSION"]["MODEL_H"])
    elif yoloVersion == "TINY":
        weightPath = config["TINY_VERSION"]["WEIGHT_PATH"]
        modelW = int(config["TINY_VERSION"]["MODEL_W"])
        modelH = int(config["TINY_VERSION"]["MODEL_H"])
    else:
        print("Version not supported, please check config file!")
        exit()
    objectDetector = PeopleDetector(weightPath, labelsPath, modelW, modelH)

    print("Waiting for image..")
    while True:
        logging.start("Get")
        frame = subRecvArray(imgSubsciber)
        h, w, _ = frame.shape
        logging.end("Get")

        logging.start("Preprocess")
        preprocessedImg = objectDetector.preprocess(frame)
        logging.end("Preprocess")

        logging.start("Inference")
        rawOutput = objectDetector.inference(preprocessedImg)
        logging.end("Inference")

        logging.start("Postprocess")
        postprocessedData = objectDetector.postprocess(rawOutput, frame, scoreThresh, nmsThresh)
        logging.end("Postprocess")
        
        logging.start("Tracking")
        outputTrackingBboxes = trackPeoples(tracker, postprocessedData)
        logging.end("Tracking")

        # Filter invalid boxes
        # newOutputTrackingBboxes = []
        # for outputTrackingBbox in outputTrackingBboxes:
        #     x0, y0, x1, y1, id = outputTrackingBbox
        #     if x0 >= 1280:
        #         x0 = 1279
        #     if y0 >= 720:
        #         y0 = 719
        #     if x1 >= 1280:
        #         x1 = 1219
        #     if y1 >= 720:
        #         y1 = 719
        #     newOutputTrackingBboxes.append([x0, y0, x1, y1, id])
        # outputTrackingBboxes = newOutputTrackingBboxes
        # outputTrackingBboxes = np.round(outputTrackingBboxes)
        # outputTrackingBboxes = outputTrackingBboxes.astype(int)

        logging.start("Send")
        pubSendArray(resultPublisher, "PEOPLE_RESULT", outputTrackingBboxes)
        logging.end("Send")

        # Print time
        logging.print_mean_result()
        


if __name__ == "__main__":
    main()
