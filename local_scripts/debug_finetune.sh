cd ..

python train.py \
--exp_id debug_0.8 \
--data data/custom.data \
--batch-size 16 \
--accumulate 1 \
--weights weights/yolov3_hand_regular_pruning_0.8percent.weights \
--cfg cfg/prune_0.8_yolov3-custom.cfg \
-sr \
--s 0.001 \
--prune 0