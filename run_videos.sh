VIDEO_DIR=/home/dmp/1.Users/1.TinhLam/4M_Videos
for folder in "$VIDEO_DIR"/*
do
  for video in "$folder"/*.avi
  do
     python3 app.py --video $video
  done
done
