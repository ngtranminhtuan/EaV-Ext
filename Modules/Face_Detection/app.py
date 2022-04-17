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


from face_detector import FaceDetector

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isFP16)

GRID = (2, 4) # 2 rows, 4 cols bouding boxes
PADDING_TOP = 30
RESIZE_FACE_RATIO = 1.7
facePaddingX = 15
facePaddingY = 15


def mergeFaces(outputTrackingBboxes, orgImg):
    filterdBoxes = []
    peopleImgs = []
    imgW, imgH = orgImg.shape[1], orgImg.shape[0]
    # Crop people bouding boxes
    mergedImg = 255 * np.ones((imgH, imgW, 3), dtype=np.uint8)
    boxIdx = 0
    for row in range(GRID[0]):
        peopleImgs.append([])
        filterdBoxes.append([])
        for col in range(GRID[1]):
            dummyImg  = np.ones((imgH//GRID[0], imgW//GRID[1], 3))
            gridW, gridH = dummyImg.shape[1], dummyImg.shape[0]
            if boxIdx >= len(outputTrackingBboxes):
                pass
            else:
                while boxIdx < len(outputTrackingBboxes):
                    person = outputTrackingBboxes[boxIdx]
                    id = person[4]
                    x0, y0, x1, y1 = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                    if (y1-y0 < 1) or (x1-x0 < 1) or (x0<0) or (y0<0) or (x1>=imgW) or (y1>=imgH):
                        boxIdx += 1
                        continue
                    else:
                        # filterdBoxes.append([x0, y0, x1, y1, id, row, col])

                        filterdBoxes[row].append([x0, y0, x1, y1, id])

                        # Crop face and add padding on top people bounding boxes
                        if y0 - PADDING_TOP >= 0:
                            peopleImg = orgImg[y0-PADDING_TOP:y0+(y1-y0)//2, x0:x1, :]
                            # filterdBoxes.append([x0, y0-PADDING_TOP, x1, y0+(y1-y0)//2])
                        else:
                            peopleImg = orgImg[0:y0+(y1-y0)//2, x0:x1, :]
                            # filterdBoxes.append([x0, 0, x1, y0+(y1-y0)//2])
                        # Resize face if size match with grid size
                        if peopleImg.shape[0]*RESIZE_FACE_RATIO < dummyImg.shape[0] and peopleImg.shape[1]*RESIZE_FACE_RATIO < dummyImg.shape[1]:
                            peopleImg = cv2.resize(peopleImg, (0,0), fx=RESIZE_FACE_RATIO, fy=RESIZE_FACE_RATIO)
                            peopleImgs[row].append(peopleImg)
                        # Crop image if size match with grid size
                        if peopleImg.shape[0] < dummyImg.shape[0] and peopleImg.shape[1] < dummyImg.shape[1]:
                            dummyImg[:peopleImg.shape[0],:peopleImg.shape[1],:] = peopleImg
                        boxIdx += 1
                        break
            mergedImg[row*gridH:(row+1)*gridH, col*gridW:(col+1)*gridW,:] = dummyImg
    return mergedImg, filterdBoxes, peopleImgs

def locateFaces(faceBoxes, orgImg, outputPeopleBoxes, peopleImgs):
    imgW, imgH = orgImg.shape[1], orgImg.shape[0]

    newFaceBoxes = []
    for boxIdx in range(len(faceBoxes)):
        for row in range(GRID[0]):
            for col in range(GRID[1]):
                if boxIdx >= len(faceBoxes):
                    break
                else:

                    dummyImg = np.ones((imgH//GRID[0], imgW//GRID[1], 3))
                    gridW, gridH = dummyImg.shape[1], dummyImg.shape[0]

                    faceBox = faceBoxes[boxIdx]
                    x0, y0, x1, y1 = faceBox[0], faceBox[1], faceBox[2], faceBox[3]

                    # Check face box in checking zone
                    if row*gridH <= y0 <= (row+1)*gridH and col*gridW <= x0 <= (col+1)*gridW:
                        # Convert from merged coordinators to crop coordinators (resized)
                        x0 -= col*gridW
                        x1 -= col*gridW
                        y0 -= row*gridH
                        y1 -= row*gridH
                    
                        if peopleImgs[row] == []:
                            continue
                        if col >= len(peopleImgs[row]):
                            continue
                    
                        # Convert from resized coordinators to cropped image coordinator
                        x0Rel = x0 / peopleImgs[row][col].shape[1]
                        x1Rel = x1 / peopleImgs[row][col].shape[1]
                        y0Rel = y0 / peopleImgs[row][col].shape[0]
                        y1Rel = y1 / peopleImgs[row][col].shape[0]

                        peopleImgs[row][col] = cv2.resize(peopleImgs[row][col], (0,0), fx=1/RESIZE_FACE_RATIO, fy=1/RESIZE_FACE_RATIO)
                        x0 = int(x0Rel * peopleImgs[row][col].shape[1])
                        x1 = int(x1Rel * peopleImgs[row][col].shape[1])
                        y0 = int(y0Rel * peopleImgs[row][col].shape[0])
                        y1 = int(y1Rel * peopleImgs[row][col].shape[0])

                        # Convert relative coordinators to absolute coordinators in original image
                        # if boxIdx < len(outputPeopleBoxes):
                        peopleBox = outputPeopleBoxes[row][col]
                        x0P = peopleBox[0] 
                        y0P = peopleBox[1]
                        # x1P = peopleBox[2]
                        # y1P = peopleBox[3]
                        id  = peopleBox[4]
                        
                        # Remove padding
                        if y0P - PADDING_TOP >= 0:
                            y0P -= PADDING_TOP
                        else:
                            y0P = 0

                        x0Absolute = int(x0 + x0P) 
                        x1Absolute = int(x1 + x0P)
                        y0Absolute = int(y0 + y0P)
                        y1Absolute = int(y1 + y0P)

                        newFaceBoxes.append([x0Absolute, y0Absolute, x1Absolute, y1Absolute, id])

                        # orgImg = rectangle_with_text(
                        #     orgImg, "", (x0Absolute, y0Absolute), (x1Absolute, y1Absolute), (255,0,0), 2
                        # )
                        boxIdx += 1
    return newFaceBoxes, orgImg

def trackFaces(tracker, bboxes, isDisabled=False):
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

def replaceAnimeFaces(orgImg, newFaceBoxes, animeImgs):
    for newFaceBox in newFaceBoxes:
        x0, y0 = newFaceBox[0]-facePaddingX, newFaceBox[1]-facePaddingY
        x1, y1 = newFaceBox[2]+facePaddingX, newFaceBox[3]+facePaddingY
        id     = newFaceBox[4]

        # Check valid faces bounding boxes
        if x0<0 or y0<0 or x1>=orgImg.shape[1] or y1>=orgImg.shape[0]:
            continue

        faceIdx  = id % 10
        animeImg = animeImgs[faceIdx]
        animeImg = cv2.resize(animeImg, (x1-x0, y1-y0))

        tempImg = orgImg.copy()
        tempImg[y0:y1,x0:x1,:] = animeImg
        mask = np.full((orgImg.shape[0], orgImg.shape[1], 1), 0, dtype=np.uint8)
        cv2.ellipse(mask , ((int((x0 + x1)/2), int((y0 + y1)/2)), (x1-x0, y1-y0), 0), 255, -1)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg  = cv2.bitwise_and(orgImg, orgImg, mask=mask_inv)
        img2_fg  = cv2.bitwise_and(tempImg,tempImg, mask=mask)
        orgImg   = cv2.add(img1_bg, img2_fg)
    return orgImg

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
    imgSubsciber = initSubsciber(subImgPort, "FACE_IMG")
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
    objectDetector = FaceDetector(weightPath, labelsPath, modelW, modelH)

    print("Waiting for image..")
    while True:
        logging.start("Get")
        frame = subRecvArray(imgSubsciber)
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
        outputTrackingBboxes = trackFaces(tracker, postprocessedData)
        logging.end("Tracking")

        logging.start("Send")
        pubSendArray(resultPublisher, "FACE_RESULT", outputTrackingBboxes)
        logging.end("Send")

        # Print time
        logging.print_mean_result()
        


if __name__ == "__main__":
    main()
