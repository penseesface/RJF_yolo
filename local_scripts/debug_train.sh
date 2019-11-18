cd ..

python train.py \
--exp_id debug \
--data data/wuxiv3.data \
--batch-size 16 \
--accumulate 1 \
--weights weights/yolov3.weights \
--cfg cfg/yolov3-custom.cfg