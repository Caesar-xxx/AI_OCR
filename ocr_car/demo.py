"""
路面车辆分析
"""
# 导入相关包
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# 导入自定义模块
from highway_detection import Detector,Tracker
import tools.infer.utility as utility
from tools.infer.predict_system import predict_rec



class CouseDemo:
    def __init__(self):
        self.frame_w = 0
        self.car_num_dict = {}
        self.right_num_info = []
        self.detect_ids = []
        args = utility.parse_args()
        
        args.det_model_dir="./weights/det/ch_db_mv3_inference/"
        args.rec_model_dir="./weights/rec/ch_ppocr_server_v2.0_rec_infer/"
        # args.rec_model_dir="./weights/rec/rec_chinese_lite_v2.0_inference/"
        args.rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls=False 
        args.use_gpu=True
        # 单独识别
        self.text_recognizer = predict_rec.TextRecognizer(args)

        self.frame_car_num = 0
        self.frame_plate_num = 0
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

    def plot(self,frame,tracker_bboxes):
        """
        绘制画面
        拿到所有的车牌未绘制的原图
        绘制车体、车牌
        绘制车牌信息
        绘制其他信息
        """
        plate_img = {}
        
        car_info_list = [t for t in tracker_bboxes if t[4] == 'car']
        plate_info_list = [t for t in tracker_bboxes if t[4] == 'plate']
        
        self.frame_car_num = len(car_info_list)
        self.frame_plate_num = len(plate_info_list)
        # 原始车牌信息
        for (l, t, r, b, cls_name, track_id) in plate_info_list:
            # 设置宽度
            if r-l > 100:
                crop = frame[t:b,l:r].copy()
                res = self.text_recognizer([crop])
                if len(res) > 0:
                    car_num,conf= res[0][0]
                    if len(car_num) == 7 and conf > 0.8:
                        # 记录最新的车牌信息
                        self.car_num_dict[track_id] = [car_num,conf]
                        if car_num not in self.detect_ids:
                            self.right_num_info.append([crop,car_num])
                            self.detect_ids.append(car_num)
                plate_img[track_id] = {'frame':crop,'box':(l, t, r, b)}
        # 绘制车身
        for (l, t, r, b, cls_name, track_id) in car_info_list:
            color = (0,255,0)
            alpha = 0.8
            frame[t:b,l:r,0] = frame[t:b,l:r,0] * alpha + color[0] * (1-alpha)
            frame[t:b,l:r,1] = frame[t:b,l:r,1] * alpha + color[1] * (1-alpha)
            frame[t:b,l:r,2] = frame[t:b,l:r,2] * alpha + color[2] * (1-alpha)
            cv2.rectangle(frame, (l,t), (r,b), (0,255,0),5)
            
        # 绘制车牌
        for (l, t, r, b, cls_name, track_id) in plate_info_list:
            cv2.rectangle(frame, (l,t), (r,b), (255,0,255),5)
        
        # 绘制浮层
        for key,plate in plate_img.items():
            plate_crop_img = plate['frame']
            l, t, r, b = plate['box']

            if key in self.car_num_dict:
                plate_crop_img = cv2.resize(plate_crop_img, (450,150))
                plate_h,plate_w = plate_crop_img.shape[:2]

                line_w,line_h = 200,200
                info_w,info_h = 700,500
                
                info_l,info_t = l - line_w - info_w,t - line_h - info_h
                info_r,info_b = info_l+info_w,info_t + info_h
                # 划线
                cv2.line(frame, (l,t), (l-line_w,t-line_h), (255,0,255),5)

                if info_l <= 0:
                    info_r = info_r-info_l
                    info_l = 0
                if info_t <= 0:
                    info_t = 0
                # 黑色底图
                color = (0,0,0)
                alpha = 0.3
                frame[info_t:info_b,info_l:info_r,0] = frame[info_t:info_b,info_l:info_r,0]  * alpha + color[0] * (1-alpha)
                frame[info_t:info_b,info_l:info_r,1] = frame[info_t:info_b,info_l:info_r,1]  * alpha + color[0] * (1-alpha)
                frame[info_t:info_b,info_l:info_r,2] = frame[info_t:info_b,info_l:info_r,2]  * alpha + color[0] * (1-alpha)

                cv2.rectangle(frame, (info_l,info_t), (info_r,info_b), (255,0,255),5)

                # 覆盖车牌
                frame[info_t:info_t+plate_h,info_l:info_l+plate_w] = plate_crop_img
                cv2.rectangle(frame, (info_l,info_t), (info_l+450,info_t+150), (255,0,255),3)
                # 文字
                car_num,conf = self.car_num_dict[key]
                cv2.putText(frame, "Id: {}".format(key), (info_l+50,info_t+plate_h+100), cv2.FONT_HERSHEY_PLAIN, 6, (0,255,0),5)
                frame = self.cv2AddChineseText(frame, str(car_num), (info_l+50,info_t+plate_h+95), (0,255,0),80)
                cv2.putText(frame, str(round(conf*100,2))+'%', (info_l+50,info_t+plate_h+280), cv2.FONT_HERSHEY_PLAIN, 6, (0,255,0),5)

        # 左上角文字    
        color = (0,0,0)
        alpha = 0.3
        l,t = 100,200
        r,b = l+500,t +400
        frame[t:b,l:r,0] = frame[t:b,l:r,0]  * alpha + color[0] * (1-alpha)
        frame[t:b,l:r,1] = frame[t:b,l:r,1]  * alpha + color[0] * (1-alpha)
        frame[t:b,l:r,2] = frame[t:b,l:r,2]  * alpha + color[0] * (1-alpha)
        
        car_num_text = '车：{}'.format(self.frame_car_num)
        frame = self.cv2AddChineseText(frame, car_num_text, (l+50,t+50), (0,255,0),100)

        plate_num_text = '车牌：{}'.format(self.frame_plate_num)
        frame = self.cv2AddChineseText(frame, plate_num_text, (l+50,t+180), (0,255,0),100)

        if len(self.right_num_info) >= 4:
            self.right_num_info = self.right_num_info[-4:]

        # 右上角信息
        l,t,r,b = -(450+120),200,-99,1500
        frame[t:b,l:r,0] = frame[t:b,l:r,0]  * alpha + color[0] * (1-alpha)
        frame[t:b,l:r,1] = frame[t:b,l:r,1]  * alpha + color[0] * (1-alpha)
        frame[t:b,l:r,2] = frame[t:b,l:r,2]  * alpha + color[0] * (1-alpha)
         
        for index,info in enumerate(self.right_num_info): 
            resize_h  = 150
            margin_b = 170
            t = 200 + (resize_h+margin_b) * index
            b = t + resize_h
            l = -(450+100)
            r = -100

            crop,car_num = info
            crop = cv2.resize(crop, (450,resize_h))
            
            frame[t:b,l:r] = crop
            frame = self.cv2AddChineseText(frame, car_num, (self.frame_w+l,b+10), (0,255,0),80)
        
        return frame
            
    def detect(self):
        # 读取视频流
        cap = cv2.VideoCapture('./videos/test2.mp4')
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧数FPS
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        # 实例化检测器
        yolo_detector = Detector()
        # 实例化2个追踪器
        tracker_left = Tracker()
        
        videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (self.frame_w ,self.frame_h ))


        # 记录帧数
        frame_index = 0
        while True:
            ret,frame = cap.read()
            
            if frame is None:
                break


            # 目标检测
            yolo_bboxes = yolo_detector.yolo_detect(frame)
            # 目标追踪
            tracker_bboxes = tracker_left.update_tracker(frame, yolo_bboxes)


            # 绘制结果
            frame = self.plot(frame, tracker_bboxes)

           
            

            # 显示
            videoWriter.write(frame)
            frame= cv2.resize(frame, (608,1080))
            cv2.imshow('demo',frame)

            frame_index +=1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        videoWriter.release()       
        cap.release()
        cv2.destroyAllWindows()


course = CouseDemo()

course.detect()