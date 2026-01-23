import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pyautogui

import cv2
import threading
import time, os
import sys
import ctypes



class OCRDetect:
    def __init__(self, setting=None, logger=None):

        self.logger = logger
        if logger is None:
            import logging
            self.logger = logging.getLogger('OCRDetect')

        if setting is None:
            setting = {'GPU': True, "time_skip": 0}

        # When frozen (PyInstaller), __file__ points inside _internal; use exe directory as app root.
        cur_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

        def _abs_if_relative(p):
            if not p:
                return p
            return p if os.path.isabs(p) else os.path.join(cur_dir, p)

        use_gpu = setting['GPU'] if 'GPU' in setting else False
        if use_gpu is True:
            self.logger.info("Using GPU checking.")
            gpu_num = self.get_gpu_count()
            if gpu_num == 0:
                use_gpu = False
            elif not self._gpu_runtime_available():
                msg = (
                    "GPU requested but CUDA runtime DLLs not found in PATH. "
                    "Paddle GPU will likely fail until CUDA/cuDNN is installed/configured."
                )
                if bool(setting.get("gpu_fallback_to_cpu", False)):
                    self.logger.warning(msg + " Falling back to CPU because settings['gpu_fallback_to_cpu']=true.")
                    use_gpu = False
                else:
                    self.logger.error(msg + " (No CPU fallback; set settings['gpu_fallback_to_cpu']=true to enable.)")

        # 首先要导入一个全局的模型，不然每次都导入，会花费额外的时间
        self.OCR_MDOEL = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, rec_image_shape='3, 24, 160', rec_batch_num=8,
                        precision='fp32', show_log=setting['log'] if 'log' in setting else False,

                       det_model_dir=_abs_if_relative(setting['det']) if 'det' in setting else os.path.join(cur_dir, 'whl', 'det', 'ch', 'ch_PP-OCRv4_det_infer'),
                       rec_model_dir=_abs_if_relative(setting['rec']) if 'rec' in setting else os.path.join(cur_dir, 'whl', 'rec', 'ch', 'ch_PP-OCRv4_rec_infer'),
                       cls_model_dir=_abs_if_relative(setting['cls']) if 'cls' in setting else os.path.join(cur_dir, 'whl', 'cls', 'ch_ppocr_mobile_v2.0_cls_infer'),
                                   )  # need to run only once to download and load model into memory
        print(os.path.join(cur_dir, 'whl/cls/ch_ppocr_mobile_v2.0_cls_infer'))
        self.time_skip = setting['time_skip'] if 'time_skip' in setting else 0

        self._measure_lock = threading.Lock()
        self._health_lock = threading.Lock()
        now = time.time()
        self._last_capture_ok_ts = now
        self._last_ocr_ok_ts = now
        self._last_loop_ok_ts = now
        self._consecutive_failures = 0
        self._last_error = None

        # 测量，缩放，是否冻结等尺度相关
        self.MEASSURE = {'增益': None, '深度': None, '频率': None, '图像增强': None,

                    'skin_distance': None, 'A': None, 'B': None, 'Alpha': None, 'Zoom_scaler': 1.0, 'Is_Freeze': False}
        self.MEASSURE['Points_Per_MM'] = None


    def get_gpu_count(self):
        import pynvml
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"可用的 GPU 数量: {count}")

            # 打印每个 GPU 的信息
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU {i}: {info}")
                self.logger.info(f"GPU {i}: {info}")

            pynvml.nvmlShutdown()
            return count
        except pynvml.NVMLError as e:
            print(f"NVML 错误: {e}")
            return 0

    def _gpu_runtime_available(self):
        # NVML only indicates GPU presence; Paddle GPU mode also requires CUDA runtime DLLs in PATH on Windows.
        if os.name != "nt":
            return True

        candidates = [
            # CUDA 12 (post120)
            "cublas64_12.dll",
            "cublas64_120.dll",
            # Other common deps
            "cudart64_12.dll",
            # cuDNN 9.x commonly ships as these on Windows
            "cudnn64_9.dll",
        ]
        for dll in candidates:
            try:
                ctypes.WinDLL(dll)
                return True
            except Exception:
                continue
        return False


    def showimg(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
    def results_show(self, img, result):
        # 显示结果
        result = result[0]
        boxes = []
        txts = []
        scores = []
        if result is not None:
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
        im_show = draw_ocr(img, boxes, txts, scores, font_path='./fonts/simfang.ttf')
        cv2.imshow('img1', im_show)
        cv2.waitKey(0)
    def find_is_freeze_in_ocr_results(self, results):
        ## 超声图像中的位置
        # 是否冻结：冻结的话，有一个雪花的标志，会被识别到一个 * 符号
        freeze_pos_col_lu = 1528 - 1304 # 286；該results的image相對於全尺寸的img的位置
        freeze_pos_row_lu = 800 - 822 # = 46；該results的image相對於全尺寸的img的位置
        freeze_pos_col_rb = 1660 - 1304  # 286；該results的image相對於全尺寸的img的位置
        freeze_pos_row_rb = 938 - 822  # = 46；該results的image相對於全尺寸的img的位置

        freeze_text = ["*", "米", "焦深", "管宽"]

        for batch_id in range(len(results)):
            batch = results[batch_id]
            if batch is None:
                return False

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                if text in freeze_text:# and freeze_pos_col_lu <= positions[0][0] and freeze_pos_row_lu <= positions[0][1] and freeze_pos_col_rb >= positions[1][0] and freeze_pos_row_rb >= positions[-1][1]:#and freeze_pos_col >= positions[0][0] and freeze_pos_col <= positions[1][0] and freeze_pos_row >= positions[0][1] and freeze_pos_row <= positions[-1][1]:
                    return False

        return True

    def detect_distance_in_img(self, img):
        ##
        # A_img = img[152:180, 1581:1704]
        # A_img = img[194:217, 1581:1704]

        A_row_start = 152
        A_row_end = 217
        A_col_start = 1555
        A_col_end = 1704

        B_row_start = 152 # 先假定在上面
        B_row_end = 180 # 先假定在上面

        B_col_start = 1743 # fixed; 冒号之后
        B_col_end = 1836 # fixed

        alpha_row_start = B_row_start
        alpha_row_end = B_row_end
        alpha_col_start = 1874
        alpha_col_end = 1907


        A_img = img[A_row_start:A_row_end, A_col_start:]

        # self.showimg(A_img)

        results = self.OCR_MDOEL.ocr(A_img)

        batch = results[0] if results else None  # 输入的图片就一张，batchsize=1
        if not batch:
            return {}

        parsed = {}
        for i in range(len(batch)):  # 每张图片中识别的所有list
            text = batch[i][1][0]
            text = text.replace('：', ':').replace('，', ',')

            if 'A:' in text and 'mm' in text:
                try:
                    start_index = text.index("A:") + 2
                    end_index = text.index("mm")
                    parsed["A"] = float(text[start_index:end_index])
                except Exception:
                    pass

            if 'B:' in text and 'mm' in text:
                try:
                    start_index = text.index("B:") + 2
                    end_index = text.rindex("mm")
                    parsed["B"] = float(text[start_index:end_index])
                except Exception:
                    pass

            if '距离' in text and 'mm' in text:
                try:
                    start_index = text.index(":") + 1
                    end_index = text.index("mm")
                    parsed['skin_distance'] = float(text[start_index:end_index])
                except Exception:
                    pass

            if ":" in text:
                try:
                    start_index = text.rindex(":") + 1
                    if start_index > 15:  # 是皮肤的距离:已经识别到;
                        alpha_text = text[start_index:].strip()
                        alpha_text = alpha_text.replace('°', '').replace('º', '')
                        if alpha_text:
                            parsed["Alpha"] = float(alpha_text)
                except Exception:
                    pass

        return parsed


        # results_show(A_img, results)


    def find_Zoom_Scaler_in_ocr_results(self, results):
        ## 超声图像中的位置
        # HIFU模式下的测试距离
        # x = (447 + 646) / 2
        # y = (92 + 115) / 2
        target_text = "缩放倍数"

        for batch_id in range(len(results)):
            batch = results[batch_id]

            if batch is None:
                return 1.0

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                text = text.replace("：", ":")
                if target_text in text:
                    # if x >= positions[0][0] and x <= positions[1][0] and y >= positions[0][1] and y <= positions[-1][1]:

                    startid = text.index(':') + 1 #英文的冒号

                    return float(text[startid:])
        return None # 没识别到

    def find_text_at_designated_postion_in_ocr_results(self, results, target_positions=[0,0]): # 找到给定position下方的数值

        position = [0, 0]
        position[0] = (target_positions[0][0] + target_positions[1][0]) // 2
        position[1] = target_positions[-1][1] + 20  # 矩形的最后一个点的y，或者第三个都行；加上的数值，需要>=15 and <=20

        for batch_id in range(len(results)):
            batch = results[batch_id]

            if batch is None:
                return None

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                # 下面矩形，右侧竖直线 穿过上面的矩形                或者              下面矩形，左侧的数值先，穿过上面的矩形
                if target_positions[0][0] < positions[1][0] < target_positions[1][0] or target_positions[0][0] < positions[0][0] < target_positions[1][0]:
                    if  positions[0][1] <= position[1] <= positions[-1][1]:
                        return text
        return None

    def find_other_setting_in_ocr_results(self, results):
        # 搞一个全局的setting，实时的更新这些数值，可能随时会被用到
        SETTINGS = {'增益': None, '深度': None, '频率': None, '图像增强': None}  # units: dB, cm, MHz        '频率': None, 'MHz': None,

        # {'增益': '60', '深度': '6.3', '频率': '8.3', '图像增强': None}; 图像增强没有识别到，因为下方的数字的框太小了
        # 增益， 深度， 动态范围，频率， 图像增强， 灰阶图谱，
        for batch_id in range(len(results)):
            batch = results[batch_id]
            if batch is None:
                return SETTINGS

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]



                for set_text in SETTINGS.keys():
                    if set_text in text: # 说明找到了需要的设置，这些text的下面就是要获取的数值
                        # if '深度' in text:
                        #     print(positions)
                        value = self.find_text_at_designated_postion_in_ocr_results(results, positions)
                        # value may be None
                        SETTINGS[set_text] = value

                        if value is not None:
                            try:
                                value = float(value)
                                SETTINGS[set_text] = value
                            except:
                                SETTINGS[set_text] = None

                            # SETTINGS[set_text] = float(value)
        return SETTINGS
    def cal_points_per_mm(self, deepth, zoom_scaler):
        # 计算单位mm，的像素个数
        # print(deepth, zoom_scaler)
        if deepth is None or zoom_scaler is None:
            return None

        num = 1 / (deepth * 10 / zoom_scaler / 734)  # 注意，734的获取方式：利用mitk或者图像软件，在刻度0的地方点击一下获得y，然后在最后一个刻度，获得坐标，然后相减即可。不用+1，因为用间隔

        return int(num + 0.5)

    def ocr_instant(self, img=None):
        # 目前只需要识别四个数值，保证实时性: skin deepth, A , B ,alpha

        # en ch

        # 实际使用的时候，需要放开以下两行
        if img is None:
            img = pyautogui.screenshot(allScreens=False, region=(0, 0, 1920, 1080))
            img = np.array(img)

        if img is None or getattr(img, "shape", None) is None:
            self.logger.warning("OCR screenshot returned empty image.")
            with self._health_lock:
                self._consecutive_failures += 1
                self._last_error = "screenshot returned empty image"
            return

        h, w = img.shape[:2]
        if h < 944 or w < 1304:
            self.logger.warning(f"OCR screenshot too small: shape={img.shape}; expected at least (944,1304).")
            with self._health_lock:
                self._consecutive_failures += 1
                self._last_error = f"screenshot too small: shape={img.shape}"
            return

        with self._health_lock:
            self._last_capture_ok_ts = time.time()

        # showimg(img)


        # time_in = time.time()

        # 2025年6月17日byYoung;需要知道是否在进行段落的测量


        updates = {}

        # 如果在段落测量，就更新AB和Alpha的值
        try:
            ab_alpha = self.detect_distance_in_img(img)
            for k, v in ab_alpha.items():
                if v is not None:
                    updates[k] = v
        except Exception:
            self.logger.exception("detect_distance_in_img failed")

        try:
            results = self.OCR_MDOEL.ocr(img[822:944, 1304:])
        except Exception:
            self.logger.exception("PaddleOCR.ocr failed")
            with self._health_lock:
                self._consecutive_failures += 1
                self._last_error = "PaddleOCR.ocr failed"
            return

        with self._health_lock:
            self._last_ocr_ok_ts = time.time()
            self._consecutive_failures = 0
            self._last_error = None

        # self.results_show(img[822:944, 1304:], results)

        zoom_scaler = self.find_Zoom_Scaler_in_ocr_results(results)
        is_freeze = self.find_is_freeze_in_ocr_results(results)
        settings = self.find_other_setting_in_ocr_results(results)

        updates['Is_Freeze'] = is_freeze
        if zoom_scaler is not None:
            updates['Zoom_scaler'] = zoom_scaler



        # MEASSURE.update(settings)# 不能直接update，因为用户可能切换，使得不是所有的setting都能识别，如深度，但是深度已经设定好了
        for key in settings.keys():
            if settings[key] is not None:  # 设定的参数，由于有移动，没有识别到，默认为前一次的结果
                updates[key] = settings[key]
            else:
                pass


        with self._measure_lock:
            deepth = self.MEASSURE.get('深度')
            zoom_scaler = updates.get('Zoom_scaler', self.MEASSURE.get('Zoom_scaler'))

        points_per_mm = self.cal_points_per_mm(deepth, zoom_scaler)
        if points_per_mm is not None:
            updates['Points_Per_MM'] = points_per_mm

        with self._measure_lock:
            for k, v in updates.items():
                self.MEASSURE[k] = v

        with self._health_lock:
            self._last_loop_ok_ts = time.time()

        # time_in = time.time() - time_in
        # print(time_in)


    def start_ocr_server(self):
        while True:
            try:
                self.ocr_instant()
                time.sleep(self.time_skip)
            except Exception:
                self.logger.exception("Unhandled error in OCR loop")
                time.sleep(0.5)

    def stop_ocr_server(self):
        pass

    def get_measures(self):
        with self._measure_lock:
            return dict(self.MEASSURE)

    def get_health(self):
        with self._health_lock:
            return {
                "last_capture_ok_ts": self._last_capture_ok_ts,
                "last_ocr_ok_ts": self._last_ocr_ok_ts,
                "last_loop_ok_ts": self._last_loop_ok_ts,
                "consecutive_failures": self._consecutive_failures,
                "last_error": self._last_error,
            }



if __name__ == '__main__':

    # for i in range(9):
    #     path = f'screensshots/{i}.bmp'
    #     img = cv2.imread(path)
    #     ocr_instant(img)

    path = f'screensshots/new1.bmp'
    img = cv2.imread(path)

    # print(img.shape)
    cv2.imshow("img1", img)
    cv2.waitKey(0)
    print(img.shape)

    time_in1 = time.time()
    ocr = OCRDetect()

    time_in2 = time.time()

    counter = 100
    while counter > 0:
        time_in3 = time.time()
        # print(time_in2 - time_in1)

        ocr.ocr_instant()

        print(time.time() - time_in3)
        print(ocr.MEASSURE)
        counter -= 1

    print(time.time() - time_in2)





