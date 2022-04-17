import os, sys
sys.path.append("../..")

from Utils.Performance_Measurement.app import Logging
from Utils.Config.app import getConfig
from Utils.Cv2_Effect.app import *
from Utils.Gpu.app import splitGPU, initFP16, initXLA


# Tensorflow 1
# import tensorflow as tf

# Tensorflow 2
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import numpy as np
import cv2
import argparse
import glob
import colorsys
import os

import utils as utils
from style_transfer import Transfer
from style_transferer import *
from Utils.Gpu.app import splitGPU, initFP16, initXLA

import random
import time
import threading

tf.compat.v1.disable_eager_execution()

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isXLA)

i = 0
styleTransfer = None

def main():

    global styleTransfer

    # Init logging to measure performance
    logging = Logging(False)

    # Get config vars
    vRamUsage = int(config["PERFORMANCE"]["VRAM"])
    modelWidth = int(config["MODEL_CONFIG"]["MODEL_W"])
    modelHeight = int(config["MODEL_CONFIG"]["MODEL_H"])

    # Gan Model
    weightPath1 = config["MODEL_CONFIG"]["WEIGHT_PATH_1"]
    weightPath2 = config["MODEL_CONFIG"]["WEIGHT_PATH_2"]
    weightPath3 = config["MODEL_CONFIG"]["WEIGHT_PATH_3"]
    weightPath4 = config["MODEL_CONFIG"]["WEIGHT_PATH_4"]
    weightPath5 = config["MODEL_CONFIG"]["WEIGHT_PATH_5"]
    weightPath6 = config["MODEL_CONFIG"]["WEIGHT_PATH_6"]
    weightPath7 = config["MODEL_CONFIG"]["WEIGHT_PATH_7"]
    weightPath8 = config["MODEL_CONFIG"]["WEIGHT_PATH_8"]
    weightPath9 = config["MODEL_CONFIG"]["WEIGHT_PATH_9"]
    weightPath10 = config["MODEL_CONFIG"]["WEIGHT_PATH_10"]
    weightPath11 = config["MODEL_CONFIG"]["WEIGHT_PATH_11"]

    weightPathList = [  weightPath1, weightPath2, weightPath3, weightPath4,
                        weightPath5, weightPath6, weightPath7, weightPath8,
                        weightPath9, weightPath10, weightPath11]

    # Split GPU
    splitGPU(vRamUsage)

    # Init pusher to push img to features
    subImgPort = int(config["COMMUNICATION"]["SUB_IMG_PORT"])
    imgSubsciber = initSubsciber(subImgPort, "ORG_IMG")
    pubResultPort = int(config["COMMUNICATION"]["PUB_RESULT_PORT"])
    resultPublisher = initPublisher(pubResultPort)
    switchStyleTime = int(config["MODEL_CONFIG"]["SWITCH_STYLE_TIME"])
    weightPath = random.choice(weightPathList)
    styleTransfer = StyleTransfer(weightPath, modelWidth, modelHeight)
  

    def initNextModel(switchStyleTime):
        styleTransferTemp = None
        previousTime = 0
        isModelLoaded = False
        global i
        global styleTransfer
 
        while True:
            # print("THREAD----------------------------------")
            # print(styleTransfer)
            # i += 1
            # time.sleep(1)
            if time.time() - previousTime > switchStyleTime - switchStyleTime//2 and not isModelLoaded:
                # print("LOADING...")
                weightPath = random.choice(weightPathList)
                styleTransferTemp = StyleTransfer(weightPath, modelWidth, modelHeight)
                dummyImg = np.zeros([720,1280,3],dtype=np.uint8)
                for i in range(3):
                    # print("INFERENCE...")
                    _ = styleTransferTemp.inference(dummyImg)
                isModelLoaded = True

            if time.time() - previousTime >= switchStyleTime:
                print("SET...")
                styleTransfer = styleTransferTemp
                previousTime = time.time()
                isModelLoaded = False

    threading.Thread(target=initNextModel, args=(switchStyleTime,), daemon=True).start()

    print("Waiting for image..")
    while True:
        # print("MAIN", i)
        # time.sleep(0.3)
        # print(styleTransfer)
        # dummyImg = np.zeros([720,1280,3],dtype=np.uint8)
        # _ = styleTransfer.inference(dummyImg)
        # continue

 
            
        logging.start("Get")
        frame = subRecvArray(imgSubsciber)
        logging.end("Get")

        logging.start("Inference")
        rawOutput = styleTransfer.inference(frame)
        logging.end("Inference")

        logging.start("Postprocess")
        result = styleTransfer.postprocess(rawOutput)
        logging.end("Postprocess")
    
        logging.start("Send")
        pubSendArray(resultPublisher, "STYLE_RESULT", result)
        logging.end("Send")

        # Print time
        logging.print_mean_result()


if __name__ == "__main__":
    main()
