from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv


config_file = '../configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco.py checkpoints/epoch_12.pth'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../checkpoints/epoch_12.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image
img = './demo.jpg'
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result)

AttributeError: 'ConfigDict' object has no attribute 'pipeline'
