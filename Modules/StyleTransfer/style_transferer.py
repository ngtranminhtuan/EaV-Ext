import os, sys
sys.path.append("../..")

from Utils.Communication.app import *
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

tf.compat.v1.disable_eager_execution()

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isXLA)

class StyleTransfer(object):

    def __init__(self, weightPath=None, width=1280, height=720):
        if weightPath:
            self.width = width
            self.height = height
            self.weightPath = weightPath
            self.loadGraph(weightPath)

    def loadGraph(self, weightPath):
        g = tf.Graph()
        config = tf.ConfigProto()
        config.graph_options.rewrite_options.auto_mixed_precision = 1

        self.sess = tf.Session(config=config)
        
        batch_shape = (None, self.height, self.width, 3)
        self.img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

        model = Transfer()
        self.pred = model(self.img_placeholder)
        
        saver = tf.train.Saver()

        if os.path.isdir(self.weightPath):
            ckpt = tf.train.get_checkpoint_state(self.weightPath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('No checkpoint found...')
        else:
            saver.restore(self.sess, self.weightPath)

    def inference(self, img):
        rawOutput = self.sess.run(self.pred, feed_dict={self.img_placeholder: np.asarray([img]).astype(np.float32)})
        return rawOutput

    def postprocess(self, rawPrediction):
        result = np.squeeze(np.clip(rawPrediction, 0, 255).astype(np.uint8))
        return result

def runVideo(logging, video, outputDir, styleTransfer):
    # Init capture
    cap = cv2.VideoCapture(video)

    # Init writer
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    basename = os.path.basename(video)
    videoWriter = cv2.VideoWriter(
        os.path.join(outputDir, "out_" + basename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        logging.start("Inference time")
        rawOutput = styleTransfer.inference(frame)
        logging.end("Inference time")

        logging.start("PostProcess time")
        result = styleTransfer.postprocess(rawOutput)
        logging.end("PostProcess time")

        if outputDir:
            videoWriter.write(frame)

        # Print time
        logging.print_mean_result()
    cap.release()
    videoWriter.release()


def runCam(logging, cam, outputDir, styleTransfer):
    # Init capture
    cap = cv2.VideoCapture(cam)
    # Init writer
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    videoWriter = cv2.VideoWriter(
        os.path.join(outputDir, "out_cam_" + str(cam) + ".mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        logging.start("Inference time")
        rawOutput = styleTransfer.inference(frame)
        logging.end("Inference time")

        logging.start("PostProcess time")
        result = styleTransfer.postprocess(rawOutput)
        logging.end("PostProcess time")

        if outputDir:
            videoWriter.write(frame)

        # Print time
        logging.print_mean_result()
    cap.release()
    videoWriter.release()


def main(args):

    # Init logging to measure performance
    logging = Logging()

    # Get config vars
    vRamUsage = int(config["PERFORMANCE"]["VRAM"])
    modelWidth = int(config["MODEL_CONFIG"]["MODEL_W"])
    modelHeight = int(config["MODEL_CONFIG"]["MODEL_H"])
    weightPath = config["MODEL_CONFIG"]["WEIGHT_PATH_7"]

    # Split GPU
    splitGPU(vRamUsage)

    # Init model
    styleTransfer = StyleTransfer(weightPath, modelWidth, modelHeight)

    if args.img:
        runImg(logging, args.img, args.output, styleTransfer)
    elif args.imgs:
        runImgs(logging, args.imgs, args.output, styleTransfer)
    elif args.video:
        runVideo(
            logging, args.video, args.output, styleTransfer
        )
    elif args.cam is not None:
        runCam(logging, args.cam, args.output, styleTransfer)
    else:
        print("Please specify image path, image dir or video path!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="", help="Path of image file")
    parser.add_argument("--imgs", type=str, default="", help="Path of image directory")
    parser.add_argument("--video", type=str, default="", help="Path of video file")
    parser.add_argument("--cam", type=int, default=None, help="Cam index")
    parser.add_argument(
        "--output", type=str, default="", help="Path of video file or cam index"
    )
    args = parser.parse_args()

    main(args)
