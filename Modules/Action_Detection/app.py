from email import message
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


from action_detector import ActionDetector

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isFP16)


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


def runActionsFilter(actionResults, previousActions):
    actionNumFilter = 5
    newActionResults = actionResults.copy()
    for id, action in actionResults.items():
        if str(id) not in previousActions:
            previousActions[str(id)] = [action]
        else:
            previousActions[str(id)].append(action)
            if len(previousActions[str(id)]) > actionNumFilter:
                previousActions[str(id)].pop(0)
        crrAction = max(previousActions[str(id)],key=previousActions[str(id)].count)
        if previousActions[str(id)].count(crrAction) / len(previousActions[str(id)]) > 0.5:
            newActionResults[id] = crrAction
        else:
            newActionResults[id] = "Analyzing"
    return newActionResults, previousActions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',default=False ,action='store_true', help='Active debug mode, use camera 0 as input')
    args = parser.parse_args()
    debugMode = args.debug

    # Init logging to measure performance
    logging = Logging(debugMode)

    # Get config vars
    poseWeight = config["DEFAULT"]["POSE_WEIGHT"]
    classificationWeight = config["DEFAULT"]["CLASSIFICATION_WEIGHT"]
    classificationClasses = config["DEFAULT"]["CLASSIFICATION_CLASSES"]
    classificationClasses = classificationClasses.replace("'", "")
    classificationClasses = classificationClasses.replace("[", "")
    classificationClasses = classificationClasses.replace("]", "")
    classificationClasses = classificationClasses.replace(" ", "")
    classificationClasses = classificationClasses.split(",")
    classificationScore = float(config["DEFAULT"]["CLASSIFICATION_SCORE"])

    # Split GPU
    vRamUsage   = int(config["PERFORMANCE"]["VRAM"])
    splitGPU(vRamUsage)
    
    # Init pusher to push img to features
    subImgPort = int(config["COMMUNICATION"]["SUB_IMG_PORT"])
    imgSubsciber = initSubsciber(subImgPort, "ORG_IMG")
    pubResultPort = int(config["COMMUNICATION"]["PUB_RESULT_PORT"])
    resultPublisher = initPublisher(pubResultPort)

    ## Init model
    actionDetector = ActionDetector(poseWeight, classificationWeight, classificationClasses)

    print("Waiting for image..")
    while True:
        logging.start("Get")
        frame = subRecvArray(imgSubsciber)
        logging.end("Get")

        logging.start("Preprocess time")
        preprocessedImg = actionDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        results = actionDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        # resultImg, middleBottomPoints = actionDetector.postprocess(frame, results)
        results = actionDetector.postprocess(frame, results)
        logging.end("Postprocess time")

        logging.start("Send")
        # pubSendArray(resultPublisher, "ACTION_RESULT", np.array([results]))
        message = {"poseKeypoints":results[0], "actionResults":results[1], "skeletons":results[2],"middleBottomPoints":results[3], "nosePoints":results[4]}
        sendZippedPickle(resultPublisher, "ACTION_RESULT", message)
        logging.end("Send")

        # Print time
        logging.print_mean_result()

        


if __name__ == "__main__":
    main()
