VIDEO_DIR=/home/lkt/2.HDD/3.Task/30.Exhibition_Demo/2.Output/1.Source/26.Find_Scale_Pose/4M_Videos
for video in "$VIDEO_DIR"/*
do
    python3 action_detector.py --video $video >> "$video".txt
    # echo $video
done
