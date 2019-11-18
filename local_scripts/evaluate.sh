cd ..

python test.py \
--cfg exp/debug_prune/pruned0.8_finetuned.cfg \
--data data/wuxiv3.data \
--weights exp/debug_prune/model_best.weights \
--batch-size 1 \
--img-size 416 \
--iou-thres 0.5