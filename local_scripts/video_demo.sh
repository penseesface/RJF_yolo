cd ..

python video_demo.py \
--data data/wuxiv2.data \
--weights exp/wuxiv2_darknet53_pruned0.6/pruned_model.weights \
--cfg exp/wuxiv2_darknet53_pruned0.6/pruned_model_0.6.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.3