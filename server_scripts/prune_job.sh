cd ..

python regular_prune.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.2 \
--cfg cfg/yolov3-custom.cfg \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline/model_best.pt \
--percent 0.2

python regular_prune.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.4 \
--cfg cfg/yolov3-custom.cfg \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline/model_best.pt \
--percent 0.4

python regular_prune.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.6 \
--cfg cfg/yolov3-custom.cfg \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline/model_best.pt \
--percent 0.6

python regular_prune.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.8 \
--cfg cfg/yolov3-custom.cfg \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline/model_best.pt \
--percent 0.8