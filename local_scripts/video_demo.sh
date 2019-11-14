cd ..

python video_demo.py \
--data data/custom.data \
--weights weights/yolov3_hand_regular_pruning_0.7percent.weights \
--cfg cfg/prune_0.7_yolov3-custom.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.3
