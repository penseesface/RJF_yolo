cd ..

python regular_prune.py \
--exp_id debug_prune \
--cfg cfg/prune_0.7_yolov3-custom.cfg \
--data data/custom.data \
--weights exp/debug/model_last.pt \
--percent 0.0