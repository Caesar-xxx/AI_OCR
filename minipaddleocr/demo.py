# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 09月 25日 12:07
文件名：demo.py
地点：changsha
"""
"""
yolov5自定义车牌识别模型+ocr文本识别模型
"""
import cv2
import numpy as np
import torch
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tools.infer.utility as utility
from tools.infer.predict_system import predict_rec

class Plate_demo:
    def __init__(self):
        # 加载目标检测模型
        self.yolo_detector = torch.hub.load('./yolov5','custom',path='./weights/car_m300_best.pt',source='local')
        # 设置置信度阈值
        self.yolo_detector.conf = 0.4


        # 实例化文字识别模型
        args = utility.parse_args()


        args.rec_model_dir = "./weights/rec/ch_ppocr_server_v2.0_rec_infer/"
        # args.rec_model_dir="./weights/rec/ch_PP-OCRv2_rec_inference/Student"
        # args.rec_model_dir="./weights/rec/rec_chinese_lite_v2.0_inference"
        args.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls = False
        args.use_gpu = True

        self.text_recognizer = predict_rec.TextRecognizer(args)

        # 添加中文
    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/MSYH.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)






    def recognize(self):
        cap = cv2.VideoCapture('./test_imgs/test2.mp4')

        while True:
            ret,frame = cap.read()

            # 判断视频是否结束
            if frame is None:
                break

            # BGR转为RGB
            img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 推理
            results = self.yolo_detector(img_rgb)
            pd = results.pandas().xyxy[0]
            # 筛选
            car_list = pd[pd['name'] == 'car'].to_numpy()
            plate_list = pd[pd['name'] == 'plate'].to_numpy()

            # 遍历每一辆车
            for car in car_list:
                l,t,r,b = car[:4].astype('int') # 浮点型转为整形
                cv2.rectangle(frame,(l,t),(r,b),(0,255,0),5)

            # 遍历每一个车牌
            for plate in plate_list:
                l,t,r,b = plate[:4].astype('int')
                cv2.rectangle(frame,(l,t),(r,b),(255,0,255),5)

                w = r-l
                # cv2.putText(frame,str(w),(l,t-50),cv2.FONT_ITALIC,5,(0,255,0),3)
                if w > 100:
                    # 车牌识别
                    plate_crop = frame[t:b,l:r]
                    res = self.text_recognizer([plate_crop])
                    if len(res) > 0:
                        car_num ,conf = res[0][0]
                        print(car_num,conf)
                        text = '{} {}%'.format(car_num,round(conf*100,2))
                        frame = self.cv2AddChineseText(frame,text,(l,t-100),(255,0,255),60)




            # 缩放画面
            frame = cv2.resize(frame,(608,1080))

            # 显示
            cv2.imshow('video',frame)

            if cv2.waitKey(10) & 0xff == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

plate = Plate_demo()
plate.recognize()