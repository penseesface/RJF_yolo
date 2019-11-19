cd ..

python video_demo.py \
--data data/wuxiv2.data \
--weights exp/mergev3_darknet53_baseline/model_best.pt \
--cfg cfg/yolov3-custom.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.3