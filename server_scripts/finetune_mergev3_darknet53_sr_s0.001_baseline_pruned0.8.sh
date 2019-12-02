cd ..

python train.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.8_finetuned \
--epoch 100 \
--accumulate 1 \
--batch-size 150 \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.8/pruned_model.weights \
--cfg exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.8/pruned_model_0.8.cfg \
--test_interval 5 \
--evolve \
--device 4,5,6,7 \
--prune 0