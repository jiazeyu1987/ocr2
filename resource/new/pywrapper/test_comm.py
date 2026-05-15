import re

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
import sys

from ultrasound_service import UltrasoundService


class YourExistingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("你的医疗工作站主程序")
        self.resize(800, 600)

        self.ultrasound_backend = UltrasoundService()

        self.ultrasound_backend.state_updated.connect(self.on_device_state_changed)
        self.ultrasound_backend.frame_received.connect(self.process_ultrasound_frame)

        self.ultrasound_backend.start_engine()

        # ====== 1. 新增：开启定时器，后台平滑获取 Provider 数据 ======
        self.current_focus_point = None
        self.provider_timer = QTimer(self)
        self.provider_timer.timeout.connect(self.poll_provider_data)
        # 每 500ms(0.5秒) 刷新一次焦点参数，避免阻塞 60FPS 的图像渲染
        self.provider_timer.start(500)

        layout = QVBoxLayout()
        self.image_label = QLabel("正在等待超声图像输入...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setMinimumSize(640, 480)

        btn = QPushButton("测试调用 fetch_provider")
        btn.clicked.connect(self.check_provider_data)

        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def poll_provider_data(self):
        """后台定时轮询最新焦点数据"""
        data_dict = self.ultrasound_backend.fetch_provider()
        if "FocusPoint" in data_dict:
            self.current_focus_point = data_dict["FocusPoint"]

    def process_ultrasound_frame(self, image_matrix):
        try:
            height, width, channels = image_matrix.shape

            # 1. 判断颜色通道并转换格式
            if channels == 4:
                image_format = QImage.Format_ARGB32
            elif channels == 3:
                image_format = QImage.Format_RGB888
            else:
                image_format = QImage.Format_Grayscale8

            # 2. 构造 QImage 并转为 QPixmap 以便绘图
            q_img = QImage(image_matrix.data, width, height, channels * width, image_format).copy()
            pixmap = QPixmap.fromImage(q_img)

            # 3. 提取焦点数据并在原图上绘制
            if getattr(self, "current_focus_point", None) is not None:
                # 预期的格式: "PointF(434.85052, 272.8398)"
                match = re.search(r'PointF\(([^,]+),\s*([^)]+)\)', self.current_focus_point)

                if match:
                    # 强转为整型像素坐标
                    x = int(float(match.group(1)))
                    y = int(float(match.group(2)))

                    # 开启原图绘制
                    painter = QPainter(pixmap)

                    # 设置黄色高亮画笔，线宽 2
                    pen = QPen(QColor(255, 255, 0))
                    pen.setWidth(2)
                    painter.setPen(pen)

                    # 绘制十字准心
                    painter.drawLine(x - 10, y, x + 10, y)
                    painter.drawLine(x, y - 10, x, y + 10)

                    painter.end()

            # 4. 将绘制好焦点的图片，自适应缩放并显示到 UI 上
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

        except Exception as e:
            print(f"PyQt 显示转换/绘制失败: {e}")

    def on_device_state_changed(self, state):
        pass

    def check_provider_data(self):
        data_dict = self.ultrasound_backend.fetch_provider()
        print(data_dict)

    def closeEvent(self, event):
        self.provider_timer.stop()  # 退出时关掉定时器
        self.ultrasound_backend.stop_engine()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YourExistingApp()
    window.show()
    sys.exit(app.exec_())