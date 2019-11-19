cd ..

python test.py \
--cfg cfg/yolov3-custom.cfg \
--data data/wuxiv3.data \
--weights exp/mergev3_darknet53_baseline/model_best.pt \
--batch-size 1 \
--img-size 416 \
--iou-thres 0.75