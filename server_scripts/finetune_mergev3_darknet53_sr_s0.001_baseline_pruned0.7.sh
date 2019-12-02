cd ..

python train.py \
--exp_id mergev3_darknet53_sr_s0.001_baseline_pruned0.7_finetuned \
--epoch 100 \
--accumulate 1 \
--batch-size 150 \
--data data/merge.data \
--weights exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.7/pruned_model.weights \
--cfg exp/mergev3_darknet53_sr_s0.001_baseline_pruned0.7/pruned_model_0.7.cfg \
--test_interval 5 \
--evolve \
--device 0,1,2,3 \
--prune 0