import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tools.infer.utility as utility
from tools.infer.predict_system import TextSystem,predict_rec


# OCR
class Baidu_PP_OCR:
    def __init__(self):

        args = utility.parse_args()
        
        args.det_model_dir="./weights/det/ch_db_mv3_inference/"
        args.rec_model_dir="./weights/rec/ch_ppocr_server_v2.0_rec_infer/"
        # args.rec_model_dir="./weights/rec/ch_PP-OCRv2_rec_inference/Student"
        # args.rec_model_dir="./weights/rec/rec_chinese_lite_v2.0_inference"
        args.rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls=False 
        args.use_gpu=True
        # 检测加识别
        self.text_sys = TextSystem(args)
        # 单独识别
        self.text_recognizer = predict_rec.TextRecognizer(args)

        # 热身
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)
    
    # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
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



    def test_detect_ocr(self):
        """
        测试检测+识别
        """
        image_dir="./test_imgs/pp.png"
        img = cv2.imread(image_dir)
        
        dt_boxes, rec_res, *rest = self.text_sys(img)
        # print(len(self.text_sys(img)))
        # print(self.text_sys(img))





     
        for index,box in enumerate(dt_boxes) :
            box = np.array(box).astype(np.int32)
            cv2.polylines(img, [box], True, color=(0, 255, 0), thickness=5)
            l,t = box[0][0],box[0][1]
            text, score = rec_res[index]
            img = self.cv2AddChineseText(img, text, (l,t-100)) 
            print(text, score)     

        cv2.imwrite('./output.jpg',img)

    def test_ocr_rec(self):
        """
        测试识别功能
        """
        image_dir="./test_imgs/test_123.jpg" 
        img = cv2.imread(image_dir)
        res = self.text_recognizer([img])
        if len(res) > 0:
            car_num,conf= res[0][0]
            print(car_num,conf)



ocr = Baidu_PP_OCR()
ocr.test_detect_ocr()
# ocr.test_ocr_rec()
