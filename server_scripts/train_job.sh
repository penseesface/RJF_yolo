cd ..

python train.py \
--exp_id wuxiv3_yolo_baseline \
--epochs 250 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights weights/yolov3.weights \
--cfg cfg/yolov3-custom.cfg

python train.py \
--exp_id wuxiv3_yolo_baseline_prune0_s0.001 \
--epochs 250 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights weights/yolov3.weights \
--cfg cfg/yolov3-custom.cfg \
-sr \
--s 0.001 \
--prune 0

python train.py \
--exp_id wuxiv3_yolo_baseline_prune1_s0.001 \
--epochs 250 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights weights/yolov3.weights \
--cfg cfg/yolov3-custom.cfg \
-sr \
--s 0.001 \
--prune 1

python train.py \
--exp_id wuxiv3_yolo_baseline_prune2_s0.001 \
--epochs 250 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights weights/yolov3.weights \
--cfg cfg/yolov3-custom.cfg \
-sr \
--s 0.001 \
--prune 2