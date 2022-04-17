VIDEO_DIR=/home/lkt/2.HDD/3.Task/30.Exhibition_Demo/2.Output/1.Source/26.Find_Scale_Pose/201911-Action-Recognition-My-Own-Dataset/source_images3
for folder in "$VIDEO_DIR"/*
do
    python3 action_detector.py --imgs $folder > "$folder".txt
    # echo $video
done
