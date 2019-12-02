cd ..

python train.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_scratch \
--accumulate 1 \
--batch-size 150 \
--data data/merge.data \
--cfg cfg/yolov3-custom.cfg \
--test_interval 5 \
--evolve \
-sr \
--s 0.001 \
--prune 0 \
--device 4,5,6,7