cd ..

python train.py \
--exp_id wuxiv2_finetuned_darknet53_percent \
--epoch 50 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights weights/yolov3_hand_regular_pruning_0.8percent.weights \
--cfg cfg/prune_0.8_yolov3-custom.cfg \
--prune 0