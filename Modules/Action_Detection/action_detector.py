import os, sys
sys.path.append("../..")

from Utils.Performance_Measurement.app import Logging
from Utils.Config.app import getConfig
from Utils.Cv2_Effect.app import *
from Utils.Gpu.app import splitGPU

# Tensorflow 2
import tensorflow as tf
import numpy as np
import cv2
import argparse
import glob

from utils.lib_tracker import Tracker
from utils.lib_classifier import *
import utils.lib_plot as lib_plot

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})



config = getConfig()


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

POSE_THRESH = 0.4

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,50,50), 4)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (57,157,221), -1)

def drawPose(frame, keypoints_with_scores, edges, confidence_threshold=0.3):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def drawAction(img_disp, dict_id2label, dict_id2skeleton):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            skeleton[1::2] = skeleton[1::2]
            lib_plot.draw_action_result(img_disp, id, skeleton, label)
    return img_disp

def remove_skeletons_with_few_joints(skeletons, imgW, imgH):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    middleBottomPoints = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xMin = 999
        xMax = -999
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)

            xList = []
            yList = []
            for i, data in enumerate(skeleton):
                if data != 0:
                    if i % 2 == 0:
                        xList.append(data)
                    else:
                        yList.append(data)

            xMin = int(min(xList) * 1280)
            xMax = int(max(xList) * 1280)

            yMin = int(min(yList) * 720)
            yMax = int(max(yList) * 720)

            bottomMiddleX = 0
            bottomMiddleY = 0
            # Save leg points
            if (skeleton[20] != 0 and skeleton[21] != 0) and (skeleton[26] != 0 and skeleton[27] != 0): 
                bottomMiddleY = int((skeleton[21] + skeleton[27]) / 2 * imgH)
                bottomMiddleX = int((skeleton[20] + skeleton[26]) / 2 * imgW)
            elif (skeleton[20] == 0 and skeleton[21] == 0) and (skeleton[26] != 0 and skeleton[27] != 0):
                bottomMiddleY = int((skeleton[27]) * imgH)
                bottomMiddleX = int((skeleton[26]) * imgW)
            elif (skeleton[20] != 0 and skeleton[21] != 0) and (skeleton[26] == 0 and skeleton[27] == 0):
                bottomMiddleY = int((skeleton[20]) * imgH)
                bottomMiddleX = int((skeleton[21]) * imgW)
            middleBottomPoints.append([bottomMiddleX, bottomMiddleY, xMin, xMax, yMin, yMax])
        # middleBottomPoints = []
    return good_skeletons, middleBottomPoints

class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        WINDOW_SIZE = 5

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)
        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]

class ActionDetector(object):

    def __init__(self, poseWeight="", classificationWeight="", classificationClasses=""):
        # Init pose (MoveNet)
        print("Loading model...")
        # model = hub.load(poseWeight)
        model = tf.saved_model.load(poseWeight)
        print("Model loaded!")
        self.movenet = model.signatures['serving_default']
        # Init Pose tracker
        self.multiperson_tracker = Tracker()
        # Init action classifier
        self.multiperson_classifier = MultiPersonClassifier(classificationWeight, classificationClasses)

    def preprocess(self, orgImg):
        tempImg = orgImg.copy()
        tempImg = tf.image.resize_with_pad(tf.expand_dims(tempImg, axis=0), 352,640)
        tempImg = tf.cast(tempImg, dtype=tf.int32)
        return tempImg

    def keypointsToSkeletons(self, keypoints_with_scores):
        skeletons = []
        for keypoints in keypoints_with_scores:
            skeleton = []
            # 0 Nose
            if keypoints[0][2] > 0.3:
                skeleton.append(keypoints[0][1])
                skeleton.append(keypoints[0][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 1 Chest (2+5)/2
            skeleton.append( (keypoints[6][1]+keypoints[5][1]) / 2 )
            skeleton.append( (keypoints[6][0]+keypoints[5][0]) / 2 )

            # 2 Right shoulder
            if keypoints[6][2] > 0.3:
                skeleton.append(keypoints[6][1])
                skeleton.append(keypoints[6][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 3 Right elbow
            if keypoints[8][2] > 0.3:
                skeleton.append(keypoints[8][1])
                skeleton.append(keypoints[8][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 4 Right wrist
            if keypoints[10][2] > 0.3:
                skeleton.append(keypoints[10][1])
                skeleton.append(keypoints[10][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 5 Left shoulder
            if keypoints[5][2] > 0.3:
                skeleton.append(keypoints[5][1])
                skeleton.append(keypoints[5][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 6 Left elbow
            if keypoints[7][2] > 0.3:
                skeleton.append(keypoints[7][1])
                skeleton.append(keypoints[7][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 7 Left wrist
            if keypoints[9][2] > 0.3:
                skeleton.append(keypoints[9][1])
                skeleton.append(keypoints[9][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 8 Right hip
            if keypoints[12][2] > 0.3:
                skeleton.append(keypoints[12][1])
                skeleton.append(keypoints[12][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 9 Right knee
            if keypoints[14][2] > 0.3:
                skeleton.append(keypoints[14][1])
                skeleton.append(keypoints[14][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 10 Right ankle
            if keypoints[16][2] > 0.3:
                skeleton.append(keypoints[16][1])
                skeleton.append(keypoints[16][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 11 Left hip
            if keypoints[11][2] > 0.3:
                skeleton.append(keypoints[11][1])
                skeleton.append(keypoints[11][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 12 Left knee
            if keypoints[13][2] > 0.3:
                skeleton.append(keypoints[13][1])
                skeleton.append(keypoints[13][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 13 Left ankle
            if keypoints[15][2] > 0.3:
                skeleton.append(keypoints[15][1])
                skeleton.append(keypoints[15][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 14 Right eye
            if keypoints[2][2] > 0.3:
                skeleton.append(keypoints[2][1])
                skeleton.append(keypoints[2][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 15 Left eye
            if keypoints[1][2] > 0.3:
                skeleton.append(keypoints[1][1])
                skeleton.append(keypoints[1][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 16 Right ear
            if keypoints[4][2] > 0.3:
                skeleton.append(keypoints[4][1])
                skeleton.append(keypoints[4][0])
            else:
                skeleton.append(0)
                skeleton.append(0)

            # 17 Left ear
            if keypoints[3][2] > 0.3:
                skeleton.append(keypoints[3][1])
                skeleton.append(keypoints[3][0])
            else:
                skeleton.append(0)
                skeleton.append(0)
            skeletons.append(skeleton)

        return skeletons

    def draw(self, img, skeletons):
        for skeleton in skeletons:
            for i in range(18):
                img = cv2.circle(img, (int(skeleton[2*i]*img.shape[1]), int(skeleton[2*i+1]*img.shape[0])), 6, (0,255,0), -1)
        return img
    

    def postprocess(self, orgImg, results):
        # tempImg = orgImg.copy()
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

        # Remove skeleton with low score
        newKeypointsWithScores = []
        for skeleton in keypoints_with_scores:
            avgScore = np.average(np.array(skeleton), axis=0)[2]
            if avgScore < POSE_THRESH:
                continue
            newKeypointsWithScores.append(skeleton)
        keypoints_with_scores = newKeypointsWithScores

        # Scale pose
        SCALE_RATIO = 1
        keypoints_with_scores_scale = []
        for skeleton in keypoints_with_scores:
            newSkeleton = []
            distanceX = 0
            distanceY = 0
            for keypoint in skeleton:
                y, x, score = keypoint
                newY, newX, = y*SCALE_RATIO, x*SCALE_RATIO
                # Get distance to move pose to middle frame
                if distanceX == 0 and distanceY == 0:
                    distanceX = (newX-0.5)
                    distanceY = (newY-0.2)
                newY, newX = (newY-distanceY), (newX - distanceX)
                newSkeleton.append([newY, newX, score])
            keypoints_with_scores_scale.append(newSkeleton)

        # Scale poses
        skeletons = self.keypointsToSkeletons(keypoints_with_scores_scale)
        skeletons, middleBottomPoints = remove_skeletons_with_few_joints(skeletons, orgImg.shape[1], orgImg.shape[0])
        dict_id2skeleton, dict_id_idx = self.multiperson_tracker.track(skeletons)

        # Original poses
        skeletons = self.keypointsToSkeletons(keypoints_with_scores)
        skeletons, middleBottomPoints = remove_skeletons_with_few_joints(skeletons, orgImg.shape[1], orgImg.shape[0])

        dict_id2label = {}
        nosePoints = []
        if len(dict_id2skeleton):
            dict_id2label = self.multiperson_classifier.classify(dict_id2skeleton)

            # Update new dict (from scale poses to original poses)
            new_dict_id2skeleton = {}
            for id, data in dict_id2skeleton.items():
                index = dict_id_idx[id]
                if id not in new_dict_id2skeleton:
                    new_dict_id2skeleton[id] = skeletons[index]
            dict_id2skeleton = new_dict_id2skeleton

            # Rule base for waving action
            for id, skeleton in dict_id2skeleton.items():
                rightEyeY       = skeleton[29]
                leftWristY  = skeleton[15]
                rightWristY = skeleton[9]
                if leftWristY == 0 or rightWristY == 0:
                    continue
                elif leftWristY < rightEyeY or rightWristY < rightEyeY:
                    dict_id2label[id] = "wave"
                    continue
                
                if dict_id2label[id] == "wave" and leftWristY > rightEyeY and rightWristY > rightEyeY:
                    dict_id2label[id] = "analyzing"


        return [keypoints_with_scores, dict_id2label, dict_id2skeleton, middleBottomPoints, nosePoints]

    def detect(self, orgImg):
        preprocessedImg = self.preprocess(orgImg)
        results = self.inference(preprocessedImg)
        resultImg = self.postprocess(orgImg, results)
        return resultImg

    def inference(self, preprocessedImg):
        results = self.movenet(preprocessedImg)
        return results



def runImg(logging, img, outputDir, actionDetector, scoreThresh):
    frame = cv2.imread(img)

    logging.start("Preprocess time")
    preprocessedImg = actionDetector.preprocess(frame)
    logging.end("Preprocess time")

    logging.start("Inference time")
    results = actionDetector.inference(preprocessedImg)
    logging.end("Inference time")

    logging.start("Postprocess time")
    resultImg = actionDetector.postprocess(frame, results)
    logging.end("Postprocess time")

    cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Output Image", resultImg)
    cv2.waitKey(0)

    if outputDir:
        basename = os.path.basename(img)
        outputPath = os.path.join(outputDir, "out_" + basename)
        cv2.imwrite(outputPath, frame)

    # Print time
    logging.print_mean_result()


def runImgs(logging, imgs, outputDir, actionDetector, scoreThresh):
    imgPaths = glob.glob(os.path.join(imgs, "*.png")) + glob.glob(
        os.path.join(imgs, "*.jpg")
    )

    if len(imgPaths) == 0:
        print("Cannot found images, please check image directory!")
        exit()
    totalDistance = []
    for imgPath in imgPaths:
        frame = cv2.imread(imgPath)

        logging.start("Preprocess time")
        preprocessedImg = actionDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        results = actionDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        results = actionDetector.postprocess(frame, results)
        logging.end("Postprocess time")

        poseKeypoints      = results[0]
        threshScore = 0.3
        for skeleton in poseKeypoints:            
            noseY, noseX, noseScore = skeleton[0]
            leftLegY, leftLegX, leftLegScore = skeleton[15]
            if noseScore > threshScore and leftLegScore > threshScore:

                xDistance = abs(leftLegX - noseX)
                yDistance = abs(leftLegY - noseY)
                totalDistance.append([xDistance, yDistance])

                drawPose(frame, [skeleton], EDGES, confidence_threshold=0.3)

        cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Output Image", frame)
        if cv2.waitKey(1) == ord("q"):
            break

        if outputDir:
            basename = os.path.basename(imgPath)
            outputPath = os.path.join(outputDir, "out_" + basename)
            cv2.imwrite(outputPath, frame)

        # Print time
        # logging.print_mean_result()
    totalDistance = np.array(totalDistance)
    print(np.mean(totalDistance, axis=0))

def runVideo(logging, video, outputDir, actionDetector, scoreThresh):
    # Init capture
    cap = cv2.VideoCapture(video)
    # Init writer
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = 20
    basename = os.path.basename(video)
    videoWriter = cv2.VideoWriter(
        os.path.join(outputDir, "out_" + basename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )

    totalDistance = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        logging.start("Preprocess time")
        preprocessedImg = actionDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        results = actionDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        results = actionDetector.postprocess(frame, results)
        logging.end("Postprocess time")

        poseKeypoints = results[0]
        newPoseKeypoints = []
        threshScore = 0.3
        dumpImg = np.zeros([240,426,3],dtype=np.uint8)

        SCALE_RATIO = 3.3
        SCALE_RATIO = 1
        for skeleton in poseKeypoints:
            newSkeleton = []
            tempImg = frame.copy()
            distanceX = 0
            distanceY = 0
            for keypoint in skeleton:
                y, x, score = keypoint
                newY, newX, = y*SCALE_RATIO, x*SCALE_RATIO
                # Get distance to move pose to middle frame
                if distanceX == 0 and distanceY == 0:
                    distanceX = (newX-0.5)
                    distanceY = (newY-0.2)
                newY, newX = (newY-distanceY), (newX - distanceX)
                newSkeleton.append([newY, newX, score])

            # drawPose(tempImg, [skeleton], EDGES, confidence_threshold=0.3)
            drawPose(tempImg, [newSkeleton], EDGES, confidence_threshold=0.3)

            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Output Image", tempImg)
            if cv2.waitKey(0) == ord("q"):
                exit()



            



            # if noseScore > threshScore and leftLegScore > threshScore:
                

            #     xDistance = abs(leftLegX - noseX)
            #     yDistance = abs(leftLegY - noseY)
            #     totalDistance.append([xDistance, yDistance])


                

        

       

        if outputDir:
            videoWriter.write(frame)

        # Print time
        # logging.print_mean_result()
    totalDistance = np.array(totalDistance)
    print(np.mean(totalDistance, axis=0))
    cap.release()
    videoWriter.release()


def runCam(logging, cam, outputDir, actionDetector, scoreThresh):
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
        preprocessedImg = actionDetector.preprocess(frame)
        logging.end("Preprocess time")

        logging.start("Inference time")
        results = actionDetector.inference(preprocessedImg)
        logging.end("Inference time")

        logging.start("Postprocess time")
        resultImg = actionDetector.postprocess(frame, results)
        logging.end("Postprocess time")

        cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Output Image", resultImg)
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
    vRamUsage = int(config["PERFORMANCE"]["VRAM"])
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
    splitGPU(vRamUsage)

    # Init model
    actionDetector = ActionDetector(poseWeight, classificationWeight, classificationClasses)

    if args.img:
        runImg(logging, args.img, args.output, actionDetector, classificationScore)
    elif args.imgs:
        runImgs(logging, args.imgs, args.output, actionDetector, classificationScore)
    elif args.video:
        runVideo(logging, args.video, args.output, actionDetector, classificationScore)
    elif args.cam is not None:
        runCam(logging, args.cam, args.output, actionDetector, classificationScore)
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
