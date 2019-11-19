cd ..

python train.py \
--exp_id wuxiv2_pruned0.7_darknet53_finetuned \
--epoch 100 \
--accumulate 1 \
--batch-size 80 \
--data data/custom.data \
--weights exp/wuxiv2_darknet53_pruned0.7/pruned_model.weights \
--cfg exp/wuxiv2_darknet53_pruned0.7/pruned_model_0.7.cfg \
--prune 0