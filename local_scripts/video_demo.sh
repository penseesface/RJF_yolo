cd ..

python video_demo.py \
--data data/custom.data \
--weights exp/debug_prune/pruned_model_0.0.weights \
--cfg exp/debug_prune/pruned_model_0.0.cfg \
--source /home/jeff/video_4.mp4 \
--view-img \
--conf-thres 0.6