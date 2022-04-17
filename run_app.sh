#!/bin/bash
./stop_app.sh
./6.Run_GAN.sh &
./1.Run_People_Detection.sh &
./2.Run_Face_Detection.sh &
./3.Run_Action_Detection.sh &
./4.Run_Streaming_Server.sh &
sleep 8m
./5.Run_Coordination.sh &
