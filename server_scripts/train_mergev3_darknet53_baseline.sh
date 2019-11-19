cd ..

python train.py \
--exp_id mergev3_darknet53_baseline \
--epochs 200 \
--accumulate 1 \
--batch-size 140 \
--data data/merge.data \
--weights exp/mergev3_darknet53_baseline/model_last.pt \
--cfg cfg/yolov3-custom.cfg \
--test_interval 5 \
--evolve
