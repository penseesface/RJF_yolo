cd ..

python detect.py \
--data data/custom.data \
--weights weights/last.pt \
--cfg cfg/yolov3-custom.cfg \
--source /home/jeff/video_4.mp4
