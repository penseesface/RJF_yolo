cd ..

python regular_prune.py \
--exp_id wuxiv2_darknet53_pruned0.6 \
--cfg cfg/yolov3-custom.cfg \
--data data/wuxiv2.data \
--weights exp/wuxiv2_yolo_baseline_prune0_s0.001/model_best.pt \
--percent 0.6