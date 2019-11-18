cd ..

python video_demo.py \
--data data/wuxiv2.data \
--weights exp/debug_prune/model_best.weights \
--cfg exp/debug_prune/pruned0.8_finetuned.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.6