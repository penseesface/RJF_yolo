cd ..

python test.py \
--weights exp/mergev3_darknet53_sr_s0.001_baseline/model_last.pt \
--cfg cfg/yolov3-custom.cfg \
--data data/wuxiv3.data \
--batch-size 1 \
--img-size 416 \
--iou-thres 0.5