cd ..

python video_demo.py \
--data data/wuxiv2.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.4/pruned_model.weights \
--cfg exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.4/pruned_model_0.4.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.3