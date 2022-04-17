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

config = getConfig()

# Init FP16
isFP16 = True if int(config['PERFORMANCE']['FP16']) == 1 else False
initFP16(isFP16)

# Init XLA
isXLA = True if int(config['PERFORMANCE']['XLA']) == 1 else False
initXLA(isFP16)


class PeopleDetector(object):

    # Static vars
    label_names = None
    colors = None
    orgW = None
    orgH = None

    def __init__(self, weightPath=None, labelsPath="", width=416, height=416):
        if weightPath:
            self.loadGraph(weightPath)
            self.inputs = self.sess.graph.get_tensor_by_name("import/inputs:0")
            self.output_bboxes = self.sess.graph.get_tensor_by_name(
                "import/output_bboxes:0"
            )
        self.loadLabels(labelsPath)
        self.loadColors()
        self.width = width
        self.height = height

    def loadColors(self):
        num_classes = len(PeopleDetector.label_names)
        hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        PeopleDetector.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
        )

    def loadLabels(self, labelsPath):
        with open(labelsPath, "r") as f:
            PeopleDetector.label_names = [line.strip() for line in f.readlines()]

    def loadGraph(self, weightPath):
        config = tf.ConfigProto()
        config.graph_options.rewrite_options.auto_mixed_precision = 1

        with tf.Graph().as_default() as graph:
            graphDef = tf.GraphDef()
            with tf.gfile.GFile(weightPath, "rb") as fp:
                graphDef.ParseFromString(fp.read())
            tf.import_graph_def(graphDef)
        self.sess = tf.Session(
            graph=graph, config=config
        )

    def calcIou(self, box1, box2):
        """
        Computes Intersection over Union value for 2 bounding boxes
        :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
        :param box2: same as box1
        :return: IoU
        """
        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        # Add small epsilon of 1e-05 to avoid division by 0
        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou

    def nms(self, raw_predictions, score_thresh=0.005, nms_thresh=0.45):
        picked_preds = []
        picked_classes = []
        filtered_results = []

        raw_predictions = np.squeeze(raw_predictions)

        # 5 elements before (coord1, coord2, coord3, coord4, score)
        num_classes = raw_predictions.shape[-1] - 5
        for i in range(num_classes):
            picked_preds_per_class = raw_predictions[
                raw_predictions[:, 5 + i] >= score_thresh
            ]
            num_picked_preds_per_class = len(picked_preds_per_class)
            if num_picked_preds_per_class > 0:
                picked_preds_per_class[:, 4] = picked_preds_per_class[
                    :, 5 + i
                ]  # Modified score for this class
                picked_preds = np.append(picked_preds, picked_preds_per_class[:, :5])
                picked_classes = np.append(
                    picked_classes, [i] * num_picked_preds_per_class
                )

        # (coord1, coord2, coord3, coord4, score)
        bbox_attrs = np.reshape(picked_preds, (-1, 5))
        picked_classes = np.array(picked_classes, dtype="int32")
        unique_classes = list(set(picked_classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = picked_classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]

                chosen_bboxes = np.append(box, [score, cls])
                filtered_results = np.append(filtered_results, chosen_bboxes)

                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([self.calcIou(box, x) for x in cls_boxes])
                iou_mask = ious < nms_thresh
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

        return np.reshape(filtered_results, (-1, 6))

    def preprocess(self, org_image):
        image = org_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, raw_predictions, org_image, score_thresh, nms_thresh):

        # Nonmax surpression
        filtered_results = self.nms(
            raw_predictions, score_thresh=score_thresh, nms_thresh=nms_thresh
        )

        if PeopleDetector.orgW and PeopleDetector.orgH:
            h_org_image, w_org_image = PeopleDetector.orgH, PeopleDetector.orgW
        else:
            h_org_image, w_org_image = org_image.shape[:2]

        final_results = []
        for result in filtered_results:
            box = result[:4]
            score = round(result[4], 2)
            label = int(result[5])
            if score < score_thresh:
                continue
            class_name = self.label_names[label]
            if class_name == "person":
                xmin = max(int(box[0] * w_org_image), 0)
                ymin = max(int(box[1] * h_org_image), 0)
                xmax = min(int(box[2] * w_org_image), w_org_image)
                ymax = min(int(box[3] * h_org_image), h_org_image)
                final_results.append([class_name, score, xmin, ymin, xmax, ymax])
        return final_results


    def inference(self, img):
        rawOutput = self.sess.run([self.output_bboxes], feed_dict={self.inputs: img})
        return rawOutput

    @staticmethod
    def drawBoxes(img, postprocessedData):
        for result in postprocessedData:
            class_name = result[0]
            score = result[1]
            xmin = result[2]
            ymin = result[3]
            xmax = result[4]
            ymax = result[5]
            color = PeopleDetector.colors[PeopleDetector.label_names.index(class_name)]
            print(color)
        
            rectangle_with_text(img, class_name.capitalize(), (xmin, ymin), (xmax, ymax), color, 2)

    @staticmethod
    def setOrgSize(imgW, imgH):
        PeopleDetector.orgW = imgW
        PeopleDetector.orgH = imgH


def runImg(logging, img, outputDir, objectDetector, scoreThresh, nmsThresh):
    frame = cv2.imread(img)

    logging.start("Preprocess time")
    preprocessedImg = objectDetector.preprocess(frame)
    logging.end("Preprocess time")

    logging.start("Inference time")
    rawOutput = objectDetector.inference(preprocessedImg)
    logging.end("Inference time")

    logging.start("Postprocess time")
    postprocessedData = objectDetector.postprocess(
        rawOutput, frame, scoreThresh, nmsThresh
    )
    logging.end("Postprocess time")

    logging.start("Draw time")
    PeopleDetector.drawBoxes(frame, postprocessedData)
    logging.end("Draw time")

    cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Output Image", frame)
    cv2.waitKey(0)

    if outputDir:
        basename = os.path.basename(img)
        outputPath = os.path.join(outputDir, "out_" + basename)
        cv2.imwrite(outputPath, frame)

    # Print time
    logging.print_mean_result()


def runImgs(logging, imgs, outputDir, objectDetector, scoreThresh, nmsThresh):
    imgPaths = glob.glob(os.path.join(imgs, "*.png")) + glob.glob(
        os.path.join(imgs, "*.jpg")
    )

    if len(imgPaths) == 0:
        print("Cannot found images, please check image directory!")
        exit()

    for imgPath in imgPaths:
        frame = cv2.imread(imgPath)

        logging.start("Preprocess time")
        preprocessedImg = objectDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        rawOutput = objectDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        postprocessedData = objectDetector.postprocess(
            rawOutput, frame, scoreThresh, nmsThresh
        )
        logging.end("Postprocess time")

        logging.start("Draw time")
        PeopleDetector.drawBoxes(frame, postprocessedData)
        logging.end("Draw time")

        cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Output Image", frame)
        if cv2.waitKey(0) == ord("q"):
            break

        if outputDir:
            basename = os.path.basename(imgPath)
            outputPath = os.path.join(outputDir, "out_" + basename)
            cv2.imwrite(outputPath, frame)

        # Print time
        logging.print_mean_result()


def runVideo(logging, video, outputDir, objectDetector, scoreThresh, nmsThresh):
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

        logging.start("Preprocess time")
        preprocessedImg = objectDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        rawOutput = objectDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        postprocessedData = objectDetector.postprocess(
            rawOutput, frame, scoreThresh, nmsThresh
        )
        logging.end("Postprocess time")

        #logging.start("Draw time")
        #PeopleDetector.drawBoxes(frame, postprocessedData)
        #logging.end("Draw time")

        #cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
        #cv2.imshow("Output Image", frame)
        #if cv2.waitKey(1) == ord("q"):
        #    break

        if outputDir:
            videoWriter.write(frame)

        # Print time
        logging.print_mean_result()
    cap.release()
    videoWriter.release()


def runCam(logging, cam, outputDir, objectDetector, scoreThresh, nmsThresh):
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

        logging.start("Preprocess time")
        preprocessedImg = objectDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        rawOutput = objectDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        postprocessedData = objectDetector.postprocess(
            rawOutput, frame, scoreThresh, nmsThresh
        )
        logging.end("Postprocess time")

        logging.start("Draw time")
        PeopleDetector.drawBoxes(frame, postprocessedData)
        logging.end("Draw time")

        cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Output Image", frame)
        if cv2.waitKey(1) == ord("q"):
            break

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
    yoloVersion = config["DEFAULT"]["VERSION"]
    vRamUsage = int(config["PERFORMANCE"]["VRAM"])
    labelsPath = config["DEFAULT"]["LABELS_PATH"]
    scoreThresh = float(config["DEFAULT"]["SCORE_THRESH"])
    nmsThresh = float(config["DEFAULT"]["NMS_THRESH"])

    # Split GPU
    splitGPU(vRamUsage)

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

    if args.img:
        runImg(logging, args.img, args.output, objectDetector, scoreThresh, nmsThresh)
    elif args.imgs:
        runImgs(logging, args.imgs, args.output, objectDetector, scoreThresh, nmsThresh)
    elif args.video:
        runVideo(
            logging, args.video, args.output, objectDetector, scoreThresh, nmsThresh
        )
    elif args.cam is not None:
        runCam(logging, args.cam, args.output, objectDetector, scoreThresh, nmsThresh)
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
