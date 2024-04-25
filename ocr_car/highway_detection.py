import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Detector:
    """
    yolo目标检测
    """

    def __init__(self):

        self.img_size = 1280
        self.conf_thres = 0.4
        self.iou_thres=0.5

        # 目标检测权重
        self.weights = 'weights/car_m300_best.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names



    # 图片预处理
    def preprocess(self, img):

        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    # 目标检测
    def yolo_detect(self, im):

        img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres )

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return pred_boxes


        


class Tracker:
    """
    deepsort追踪
    """
    def __init__(self):

        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)


    def update_tracker(self,image, yolo_bboxes):

        bbox_xywh = []
        confs = []
        clss = []

        for x1, y1, x2, y2, cls_id, conf in yolo_bboxes:

            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        #更新追踪结果
        outputs = self.deepsort.update(xywhs, confss, clss, image)
        

        bboxes2draw = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value

            bboxes2draw.append(
                (x1, y1, x2, y2, cls_, track_id)
            )
        

        return bboxes2draw



