#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleFEM配置管理器UI
提供图形化界面来配置simple_fem_config.json
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import queue
import re
import sqlite3
import sys
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
ROI2_HIGH_GRAY_MATCH_THRESHOLD = 100


def natural_sort_key(value):
    """按自然顺序排序文件名，例如 roi1_2 在 roi1_10 前面"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


class SimpleFEMConfigGUI:
    def __init__(self, root):
        self.root = root

        # 只在非全屏模式下设置窗口大小
        if root.state() != 'zoomed':
            self.root.title("SimpleFEM 配置管理器")
            self.root.geometry("1800x1000")

        # 配置文件路径：源码运行时使用模块目录，打包后使用 exe 所在目录以便持久化
        self.bundle_dir = Path(__file__).resolve().parent
        self.runtime_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else self.bundle_dir
        self.bundled_config_path = str(self.bundle_dir / "simple_fem_config.json")
        self.config_path = str(self.runtime_dir / "simple_fem_config.json")
        self.ui_state_path = str(self.runtime_dir / "config_gui_state.json")
        self.config_data = {}
        self.ui_state_save_job = None
        self.ui_state_watchers_registered = False

        # ROI可视化相关
        self.roi1_image = None
        self.roi1_photo = None
        self.current_roi1_path = None

        # Y轴缩放相关
        self.y_zoom_factor = 1.0  # Y轴缩放因子 (1.0 = 原始大小)
        self.y_min_zoom = 0.1    # 最小缩放因子 (10%)
        self.y_max_zoom = 10.0   # 最大缩放因子 (1000%)
        self.roi1_zoom_factor = tk.DoubleVar(value=1.0)  # ROI1画布缩放因子
        self.y_zoom_step = 0.1   # 每次滚轮缩放步长

        # ROI1图片导航相关
        self.current_image_sequence = []  # 当前序列的图片列表
        self.current_image_index = -1     # 当前图片在序列中的索引
        self.extract_roi1_video_button = None
        self.is_extracting_roi1_video = False
        self.roi1_video_extract_task_id = 0
        self.roi1_video_extract_result_queue = queue.Queue()
        self.roi1_video_extract_poll_job = None

        # ROI2高灰度命中扫描相关
        self.folder_image_files = []
        self.roi2_high_gray_match_paths = []
        self.roi2_high_gray_match_indices = []
        self.current_roi2_match_position = -1
        self.is_scanning_roi2_high_gray = False
        self.roi2_high_gray_scan_id = 0
        self.roi2_high_gray_match_var = tk.StringVar(value="命中: 0/0")
        self.scan_roi2_high_gray_button = None
        self.load_offline_db_button = None
        self.prev_roi2_high_gray_button = None
        self.next_roi2_high_gray_button = None
        self.toggle_roi2_neighbor_display_button = None
        self.roi2_high_gray_scan_result_queue = queue.Queue()
        self.roi2_high_gray_scan_poll_job = None
        self.roi2_compare_resize_job = None
        self.roi2_compare_zoom_factor = 1.0
        self.roi2_compare_min_zoom = 0.2
        self.roi2_compare_max_zoom = 8.0
        self.roi2_compare_zoom_step = 0.1
        self.roi2_neighbor_show_full_image_var = tk.BooleanVar(value=False)
        self.roi2_neighbor_display_mode_var = tk.StringVar(value="上/下张显示: ROI2")
        self.roi2_hem_avg_delta_var = tk.StringVar(value="0")
        self.roi2_hem_pixel_delta_var = tk.StringVar(value="0")
        self.roi2_hem_result_var = tk.StringVar(value="--")
        self.roi2_hem_result_label = None
        self.roi2_compare_threshold_hint_var = tk.StringVar(value="计数阈值: --")
        self.roi3_scan_threshold_var = tk.StringVar(value=str(ROI2_HIGH_GRAY_MATCH_THRESHOLD))
        self.roi3_scan_button_text_var = tk.StringVar(value=f"扫描ROI3>{ROI2_HIGH_GRAY_MATCH_THRESHOLD}")
        self.roi2_compare_status_var = tk.StringVar(
            value=f"请先导入图片并扫描 ROI3>{ROI2_HIGH_GRAY_MATCH_THRESHOLD} 命中结果"
        )
        self.roi2_compare_panels = []
        self.roi2_compare_mode = "scan"  # scan / offline_db
        self.offline_db_record = None
        self.offline_db_records = []
        self.current_offline_db_position = -1
        self.offline_db_path = "D:/software_data/ccwssm"

        # ROI1画布拖拽移动相关
        self.roi1_pan_offset_x = tk.IntVar(value=0)  # ROI1画布X轴平移偏移量
        self.roi1_pan_offset_y = tk.IntVar(value=0)  # ROI1画布Y轴平移偏移量
        self.is_panning = False  # 是否正在拖拽
        self.pan_start_x = 0  # 拖拽开始时的鼠标X坐标
        self.pan_start_y = 0  # 拖拽开始时的鼠标Y坐标
        self.pan_start_offset_x = 0  # 拖拽开始时的X偏移量
        self.pan_start_offset_y = 0  # 拖拽开始时的Y偏移量

        # 阈值提取测试相关
        self.threshold_lower_var = tk.IntVar(value=140)
        self.threshold_upper_var = tk.IntVar(value=255)
        self.roi2_stats_threshold_var = tk.StringVar(value="140")
        self.threshold_mask = None
        self.largest_component_mask = None
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.overlay_alpha = tk.DoubleVar(value=0.5)
        self.current_overlay_image = None
        self.current_overlay_photo = None
        self.current_roi3_coords = None
        self.current_mask_for_overlay = None

        # 统计信息显示
        self.total_pixels_var = tk.StringVar(value="0")
        self.largest_component_pixels_var = tk.StringVar(value="0")
        self.component_count_var = tk.StringVar(value="0")
        self.threshold_percentage_var = tk.StringVar(value="0.00%")
        self.roi2_avg_gray_var = tk.StringVar(value="--")
        self.roi2_above_threshold_count_var = tk.StringVar(value="--")

        # ROI3列平均灰度差值
        self.column_mean_diff_var = tk.StringVar(value="---")

        # 上一帧平均灰度值（用于计算差值）
        self.prev_frame_avg_gray = None
        self.frame_diff_var = tk.StringVar(value="--")

        # Continuous check functionality
        self.continuous_check_enabled = tk.BooleanVar(value=False)

        # 热力图相关状态变量
        self.heat_map = None
        self.heatmap_mode = False
        self.heatmap_alpha_var = tk.DoubleVar(value=0.6)
        self.continuous_heatmap_enabled = tk.BooleanVar(value=False)
        self.roi2_stats_threshold_var.trace_add('write', lambda *args: self.on_roi2_stats_threshold_changed())
        self.roi3_scan_threshold_var.trace_add('write', lambda *args: self.on_roi3_scan_threshold_changed())
        self.roi2_neighbor_show_full_image_var.trace_add('write', lambda *args: self.on_roi2_neighbor_display_mode_changed())
        self.roi2_hem_avg_delta_var.trace_add('write', lambda *args: self.on_roi2_hem_threshold_changed())
        self.roi2_hem_pixel_delta_var.trace_add('write', lambda *args: self.on_roi2_hem_threshold_changed())

        # 创建UI
        self.create_widgets()

        # 加载配置
        self.load_config()
        self.load_ui_state()
        self.register_ui_state_watchers()
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置根窗口的行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # 标题
        title_label = ttk.Label(main_frame, text="SimpleFEM 配置管理器",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # 文件操作按钮
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(file_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="重新加载", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="另存为", command=self.save_as).pack(side=tk.LEFT, padx=5)
        row += 1

        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                            sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 配置选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(row, weight=1)

        # 基础配置标签页
        self.create_basic_tab()

        # ROI配置标签页
        self.create_roi_tab()

        # 峰值检测标签页
        self.create_peak_detection_tab()

        # ROI3覆盖标签页
        self.create_roi3_override_tab()
        self.create_roi2_match_tab()

        # 默认选中ROI配置页签（索引为1）
        try:
            self.notebook.select(1)  # ROI配置是第二个页签（索引从0开始）
            print("[DEBUG] 默认选中ROI配置页签")
        except Exception as e:
            print(f"[WARNING] 设置默认页签失败: {e}")

        row += 1

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 绑定键盘事件用于ROI1图片导航
        self.root.bind('<Key>', self.on_key_press)
        self.root.bind('<d>', lambda e: self.on_key_press(e))  # 明确绑定D键
        self.root.bind('<a>', lambda e: self.on_key_press(e))  # 明确绑定A键
        self.root.bind('<Left>', lambda e: self.on_key_press(e))  # 明确绑定左箭头
        self.root.bind('<Right>', lambda e: self.on_key_press(e))  # 明确绑定右箭头

        # 确保窗口能接收键盘事件
        print("[DEBUG] 键盘事件已绑定，尝试获取窗口焦点")

        # 延迟设置焦点，确保所有组件都已创建
        self.root.after(100, self.setup_keyboard_focus)

    def setup_keyboard_focus(self):
        """设置键盘焦点"""
        try:
            # 尝试多种方式设置焦点
            self.root.focus_set()
            self.root.grab_set()

            # 尝试将焦点设置到主窗口
            self.root.focus_force()

            print("[DEBUG] 焦点设置完成")

            # 添加焦点丢失检测
            self.root.bind('<FocusIn>', lambda e: print("[DEBUG] 窗口获得焦点"))
            self.root.bind('<FocusOut>', lambda e: print("[DEBUG] 窗口失去焦点"))

        except Exception as e:
            print(f"[ERROR] 焦点设置失败: {e}")

    def create_basic_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="基础配置")

        # 创建滚动框架
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 基础配置项
        self.basic_vars = {}

        configs = [
            ("processing_mode", "处理模式", ["screen", "video"]),
            ("data_processing.save_roi1", "保存ROI1图像", "bool"),
            ("data_processing.save_roi2", "保存ROI2图像", "bool"),
            ("data_processing.save_wave", "保存波形图", "bool"),
            ("data_processing.only_delect", "仅保存有波峰的帧", "bool"),
        ]

        row = 0
        for key, label, config_type in configs:
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(scrollable_frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            elif key == "processing_mode":
                var = tk.StringVar()
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=config_type, state="readonly")
                combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
            else:
                var = tk.StringVar()
                ttk.Entry(scrollable_frame, textvariable=var).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

            self.basic_vars[key] = var
            row += 1

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    
    def create_roi_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ROI配置")

        # 创建三栏布局：左侧-ROI预览，中部-配置，右侧-直方图
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        middle_frame = ttk.Frame(frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5), pady=10)

        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        self.roi_vars = {}

        # ===== 中部：ROI参数配置 =====

        # 创建中部配置区域
        config_upper_frame = ttk.Frame(middle_frame)
        config_upper_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== 右侧：直方图分析区域 =====
        histogram_frame = ttk.Frame(right_frame)
        histogram_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ROI1配置
        row = 0
        ttk.Label(config_upper_frame, text="ROI1 大区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi1_configs = [
            ("roi_capture.default_config.x1", "X1坐标", "int"),
            ("roi_capture.default_config.y1", "Y1坐标", "int"),
            ("roi_capture.default_config.x2", "X2坐标", "int"),
            ("roi_capture.default_config.y2", "Y2坐标", "int"),
        ]

        for key, label, config_type in roi1_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # ROI1 变化会使基于 ROI3 的命中扫描结果失效
            var.trace('w', lambda *args: self.on_roi1_or_roi3_config_changed())
            row += 1

        # ROI2配置
        ttk.Label(config_upper_frame, text="ROI2 小区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi2_configs = [
            ("roi_capture.roi2_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi2_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi2_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi2_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi2_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # ROI2 变化只刷新当前 ROI2 可视化与对比显示
            var.trace('w', lambda *args: self.on_roi2_config_changed())
            row += 1

        # ROI3配置
        ttk.Label(config_upper_frame, text="ROI3 扩展区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi3_configs = [
            ("roi_capture.roi3_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi3_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi3_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi3_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi3_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # ROI3 变化会使基于 ROI3 的命中扫描结果失效
            var.trace('w', lambda *args: self.on_roi1_or_roi3_config_changed())
            row += 1

        # ===== 阈值提取测试区域 =====
        threshold_test_frame = ttk.LabelFrame(config_upper_frame, text="阈值提取测试区域", padding=10)
        threshold_test_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(15, 5), padx=5)
        row += 1

        # 阈值控制区域
        threshold_control_frame = ttk.Frame(threshold_test_frame)
        threshold_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(threshold_control_frame, text="阈值控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        # Lower threshold
        ttk.Label(threshold_control_frame, text="下限:").pack(side=tk.LEFT, padx=(0, 5))
        lower_entry = ttk.Entry(threshold_control_frame, textvariable=self.threshold_lower_var, width=8)
        lower_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Upper threshold
        ttk.Label(threshold_control_frame, text="上限:").pack(side=tk.LEFT, padx=(0, 5))
        upper_entry = ttk.Entry(threshold_control_frame, textvariable=self.threshold_upper_var, width=8)
        upper_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(threshold_control_frame, text="ROI2计数阈值:").pack(side=tk.LEFT, padx=(10, 5))
        roi2_stats_threshold_entry = ttk.Entry(
            threshold_control_frame,
            textvariable=self.roi2_stats_threshold_var,
            width=8
        )
        roi2_stats_threshold_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(threshold_control_frame, text="(0-255)").pack(side=tk.LEFT, padx=(0, 10))

        # Buttons
        ttk.Button(threshold_control_frame, text="提取", command=self.on_threshold_submit).pack(side=tk.LEFT, padx=2)
        ttk.Button(threshold_control_frame, text="清除", command=self.on_threshold_clear).pack(side=tk.LEFT, padx=2)

        # Continuous check checkbox
        continuous_check_frame = ttk.Frame(threshold_test_frame)
        continuous_check_frame.pack(fill=tk.X, pady=(5, 5))

        self.continuous_check_checkbox = ttk.Checkbutton(
            continuous_check_frame,
            text="连续检查 (加载图片时自动执行提取+最大连通域)",
            variable=self.continuous_check_enabled
        )
        self.continuous_check_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        # 连通域分析区域
        component_frame = ttk.Frame(threshold_test_frame)
        component_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(component_frame, text="连通域分析:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(component_frame, text="最大连通域", command=self.on_max_component).pack(side=tk.LEFT, padx=2)
        ttk.Button(component_frame, text="统计", command=self.on_statistics).pack(side=tk.LEFT, padx=2)

        # 统计信息区域
        stats_frame = ttk.Frame(threshold_test_frame)
        stats_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(stats_frame, text="统计信息:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="总像素:").pack(side=tk.LEFT, padx=(0, 5))
        total_pixels_label = ttk.Label(stats_frame, textvariable=self.total_pixels_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        total_pixels_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="阈值范围占比:").pack(side=tk.LEFT, padx=(0, 5))
        percentage_label = ttk.Label(stats_frame, textvariable=self.threshold_percentage_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        percentage_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="帧差值:").pack(side=tk.LEFT, padx=(0, 5))
        frame_diff_label = ttk.Label(stats_frame, textvariable=self.frame_diff_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        frame_diff_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="最大连通域像素:").pack(side=tk.LEFT, padx=(0, 5))
        largest_pixels_label = ttk.Label(stats_frame, textvariable=self.largest_component_pixels_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        largest_pixels_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="连通域数量:").pack(side=tk.LEFT, padx=(0, 5))
        component_count_label = ttk.Label(stats_frame, textvariable=self.component_count_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        component_count_label.pack(side=tk.LEFT)

        roi2_stats_frame = ttk.Frame(threshold_test_frame)
        roi2_stats_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(roi2_stats_frame, text="ROI2统计:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(roi2_stats_frame, text="平均灰度:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            roi2_stats_frame,
            textvariable=self.roi2_avg_gray_var,
            font=('Courier', 9),
            relief='sunken',
            padding=(3, 1)
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(roi2_stats_frame, text=">阈值像素:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(
            roi2_stats_frame,
            textvariable=self.roi2_above_threshold_count_var,
            font=('Courier', 9),
            relief='sunken',
            padding=(3, 1)
        ).pack(side=tk.LEFT)

        # 叠加控制区域
        overlay_frame = ttk.Frame(threshold_test_frame)
        overlay_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(overlay_frame, text="叠加控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.overlay_checkbox = ttk.Checkbutton(overlay_frame, text="显示叠加", variable=self.overlay_enabled,
                                                command=self.on_overlay_toggle)
        self.overlay_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(overlay_frame, text="透明度:").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_scale = ttk.Scale(overlay_frame, from_=0.0, to=1.0, variable=self.overlay_alpha,
                                   orient=tk.HORIZONTAL, length=100, command=self.on_alpha_change)
        self.alpha_scale.pack(side=tk.LEFT, padx=(0, 5))

        self.alpha_value_label = ttk.Label(overlay_frame, text="0.5", font=('Courier', 9))
        self.alpha_value_label.pack(side=tk.LEFT)

        # ROI3列平均灰度差值显示
        column_mean_frame = ttk.Frame(threshold_test_frame)
        column_mean_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(column_mean_frame, text="ROI3列平均灰度差值:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.column_mean_diff_label = ttk.Label(column_mean_frame, textvariable=self.column_mean_diff_var,
                                                 font=('Arial', 10), foreground='green')
        self.column_mean_diff_label.pack(side=tk.LEFT)

        # 热力图控制区域 (在叠加控制区域下方) - 已隐藏
        # heatmap_frame = ttk.Frame(threshold_test_frame)
        # heatmap_frame.pack(fill=tk.X, pady=(5, 0))
        #
        # ttk.Label(heatmap_frame, text="热力图控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        # self.heatmap_button = ttk.Button(heatmap_frame, text="热力图显示", command=self.on_heatmap_submit)
        # self.heatmap_button.pack(side=tk.LEFT, padx=2)
        # self.heatmap_clear_button = ttk.Button(heatmap_frame, text="清除热力图", command=self.on_heatmap_clear)
        # self.heatmap_clear_button.pack(side=tk.LEFT, padx=2)
        #
        # self.continuous_heatmap_checkbox = ttk.Checkbutton(
        #     heatmap_frame,
        #     text="连续热力图 (加载图片时自动执行)",
        #     variable=self.continuous_heatmap_enabled
        # )
        # self.continuous_heatmap_checkbox.pack(side=tk.LEFT, padx=(10, 5))
        #
        # ttk.Label(heatmap_frame, text="热力图透明度:").pack(side=tk.LEFT, padx=(10, 5))
        # self.heatmap_alpha_scale = ttk.Scale(heatmap_frame, from_=0.0, to=1.0, variable=self.heatmap_alpha_var,
        #                                     orient=tk.HORIZONTAL, length=100, command=self.on_heatmap_alpha_change)
        # self.heatmap_alpha_scale.pack(side=tk.LEFT, padx=(0, 5))
        #
        # self.heatmap_alpha_value_label = ttk.Label(heatmap_frame, text="0.6", font=('Courier', 9))
        # self.heatmap_alpha_value_label.pack(side=tk.LEFT)

        # 其他配置
        ttk.Label(config_upper_frame, text="其他配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        other_configs = [
            ("roi_capture.frame_rate", "采集帧率", "int"),
            ("roi3_config.save_roi3", "保存ROI3图像", "bool"),
        ]

        for key, label, config_type in other_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(config_upper_frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                var = tk.StringVar()
                ttk.Entry(config_upper_frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)

            self.roi_vars[key] = var
            row += 1

        # ===== 直方图分析区域（独立于配置序列）=====

        # 直方图标题
        histogram_title_frame = ttk.Frame(histogram_frame)
        histogram_title_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 5))

        histogram_title = ttk.Label(histogram_title_frame, text="ROI灰度直方图分析", font=('Arial', 14, 'bold'))
        histogram_title.pack(side=tk.LEFT, padx=10)

        # 直方图画布框架
        curve_frame = ttk.LabelFrame(histogram_frame, text="直方图分析区域", padding=10)
        curve_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 0))

        self.curve_canvas = tk.Canvas(curve_frame, bg='white', width=600, height=900,
                                     highlightthickness=1, highlightbackground='gray')
        self.curve_canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定鼠标滚轮事件用于Y轴缩放
        self.curve_canvas.bind("<MouseWheel>", self.on_curve_canvas_mousewheel)
        self.curve_canvas.bind("<Control-MouseWheel>", self.on_curve_canvas_mousewheel_reset)  # Ctrl+滚轮重置
        self.curve_canvas.bind("<Button-4>", self.on_curve_canvas_mousewheel)  # Linux
        self.curve_canvas.bind("<Button-5>", self.on_curve_canvas_mousewheel)  # Linux
        self.curve_canvas.bind("<Control-Button-4>", self.on_curve_canvas_mousewheel_reset)  # Linux Ctrl+滚轮重置
        self.curve_canvas.bind("<Control-Button-5>", self.on_curve_canvas_mousewheel_reset)  # Linux Ctrl+滚轮重置

        # 为曲线画布添加焦点和键盘事件
        self.curve_canvas.bind('<Button-1>', lambda e: self.curve_canvas.focus_set())  # 点击画布时获取焦点
        self.curve_canvas.bind('<Key>', self.on_key_press)  # 在画布上监听键盘事件
        self.curve_canvas.bind('<d>', lambda e: self.on_key_press(e))
        self.curve_canvas.bind('<a>', lambda e: self.on_key_press(e))

        # ===== 左侧：ROI1预览区域 =====

        # ROI1预览标题和控制
        left_title_frame = ttk.Frame(left_frame)
        left_title_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        left_title = ttk.Label(left_title_frame, text="ROI1 区域叠加预览", font=('Arial', 14, 'bold'))
        left_title.pack(side=tk.LEFT, padx=10)

        # 图片导入按钮放在标题右侧
        button_frame = ttk.Frame(left_title_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(button_frame, text="导入图片",
                  command=self.import_roi1_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除图片",
                  command=self.clear_roi1_image).pack(side=tk.LEFT, padx=5)
        self.extract_roi1_video_button = ttk.Button(
            button_frame,
            text="视频抽帧ROI1",
            command=self.extract_roi1_frames_from_video
        )
        self.extract_roi1_video_button.pack(side=tk.LEFT, padx=5)
        # 当前图片路径显示
        self.image_path_var = tk.StringVar(value="未选择图片")
        path_frame = ttk.Frame(left_frame)
        path_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 10))

        path_label = ttk.Label(path_frame, text="当前图片:", font=('Arial', 10))
        path_label.pack(side=tk.LEFT, padx=(10, 5))

        path_value = ttk.Label(path_frame, textvariable=self.image_path_var,
                              foreground='gray', font=('Arial', 9))
        path_value.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.update_roi2_high_gray_match_ui()

        # ROI1预览画布框架
        roi_frame = ttk.LabelFrame(left_frame, text="ROI1预览区域", padding=10)
        roi_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.roi_canvas = tk.Canvas(roi_frame, bg='white', width=640, height=900,
                                   highlightthickness=1, highlightbackground='gray')
        self.roi_canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定ROI1画布鼠标事件，用于显示灰度值
        self.roi_canvas.bind('<Motion>', self.on_roi_canvas_mouse_motion)
        self.roi_canvas.bind('<Leave>', self.on_roi_canvas_mouse_leave)

        # 绑定ROI1画布鼠标滚轮事件，用于缩放
        self.roi_canvas.bind("<MouseWheel>", self.on_roi_canvas_mousewheel)
        self.roi_canvas.bind("<Control-MouseWheel>", self.on_roi_canvas_mousewheel_reset)  # Ctrl+滚轮重置
        self.roi_canvas.bind("<Button-4>", self.on_roi_canvas_mousewheel)  # Linux 向上滚轮
        self.roi_canvas.bind("<Button-5>", self.on_roi_canvas_mousewheel)  # Linux 向下滚轮
        self.roi_canvas.bind("<Control-Button-4>", self.on_roi_canvas_mousewheel_reset)  # Linux Ctrl+向上滚轮重置
        self.roi_canvas.bind("<Control-Button-5>", self.on_roi_canvas_mousewheel_reset)  # Linux Ctrl+向下滚轮重置

        # 绑定ROI1画布拖拽移动事件（鼠标滚轮按下拖拽）
        self.roi_canvas.bind("<Button-2>", self.on_roi_canvas_pan_start)  # 鼠标滚轮按下
        self.roi_canvas.bind("<B2-Motion>", self.on_roi_canvas_pan_motion)  # 按住滚轮拖拽
        self.roi_canvas.bind("<ButtonRelease-2>", self.on_roi_canvas_pan_end)  # 释放滚轮

        # 绑定双击事件重置位置
        self.roi_canvas.bind("<Double-Button-1>", self.on_roi_canvas_reset_position)  # 双击左键重置位置

        # 在ROI1画布下方添加固定的像素信息显示框
        pixel_info_frame = ttk.Frame(roi_frame)
        pixel_info_frame.pack(fill=tk.X, pady=(5, 0))

        # 像素信息标签
        info_label = ttk.Label(pixel_info_frame, text="鼠标位置信息:", font=('Arial', 10, 'bold'))
        info_label.pack(side=tk.LEFT, padx=(0, 10))

        # 像素信息显示文本框
        self.pixel_info_var = tk.StringVar(value="请导入图片并在ROI1区域移动鼠标")
        self.pixel_info_label = ttk.Label(pixel_info_frame, textvariable=self.pixel_info_var,
                                         font=('Courier', 9), foreground='blue', relief='sunken',
                                         padding=(5, 2))
        self.pixel_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 底部：图例说明 (放在直方图区域下方)
        legend_container = ttk.Frame(histogram_frame)
        legend_container.pack(fill=tk.X, pady=(5, 0))

        # ROI图例
        roi_legend_frame = ttk.LabelFrame(legend_container, text="ROI区域图例", padding=5)
        roi_legend_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        roi_legend_content = ttk.Frame(roi_legend_frame)
        roi_legend_content.pack()

        ttk.Label(roi_legend_content, text="ROI区域:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        # ROI1图例（背景）
        roi1_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi1_legend.pack(side=tk.LEFT, padx=2)
        roi1_legend.create_rectangle(2, 2, 23, 13, fill='', outline='darkgreen', width=2, dash=(3, 1))
        ttk.Label(roi_legend_content, text="ROI1(背景)", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI2图例
        roi2_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi2_legend.pack(side=tk.LEFT, padx=2)
        roi2_legend.create_rectangle(2, 2, 23, 13, fill='', outline='red', width=2)
        ttk.Label(roi_legend_content, text="ROI2", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI3图例
        roi3_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi3_legend.pack(side=tk.LEFT, padx=2)
        roi3_legend.create_rectangle(2, 2, 23, 13, fill='', outline='blue', width=2, dash=(4, 2))
        ttk.Label(roi_legend_content, text="ROI3", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 交点图例
        intersection_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        intersection_legend.pack(side=tk.LEFT, padx=2)
        intersection_legend.create_oval(7, 5, 18, 11, fill='lime', outline='darkgreen')
        ttk.Label(roi_legend_content, text="中心点", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 灰度曲线图例
        curve_legend_frame = ttk.LabelFrame(legend_container, text="灰度直方图图例", padding=5)
        curve_legend_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        curve_legend_content = ttk.Frame(curve_legend_frame)
        curve_legend_content.pack()

        ttk.Label(curve_legend_content, text="直方图:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        # ROI2灰度曲线图例
        roi2_curve_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        roi2_curve_legend.pack(side=tk.LEFT, padx=2)
        roi2_curve_legend.create_line(2, 12, 23, 3, fill='red', width=2)
        ttk.Label(curve_legend_content, text="ROI2像素分布", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI3灰度曲线图例
        roi3_curve_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        roi3_curve_legend.pack(side=tk.LEFT, padx=2)
        roi3_curve_legend.create_line(2, 12, 23, 3, fill='blue', width=2)
        ttk.Label(curve_legend_content, text="ROI3像素分布", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI3列平均灰度值曲线图例
        column_legend_canvas = tk.Canvas(curve_legend_content, width=30, height=15, bg='white', highlightthickness=0)
        column_legend_canvas.pack(side=tk.LEFT, padx=2)
        # 绘制绿色虚线
        column_legend_canvas.create_line(2, 7, 28, 7, fill='green', width=2, dash=(5, 3))
        ttk.Label(curve_legend_content, text="ROI3列平均", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 移除阈值线图例 - 灰度直方图不需要显示阈值线
        # threshold_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        # threshold_legend.pack(side=tk.LEFT, padx=2)
        # threshold_legend.create_line(2, 8, 23, 8, fill='green', width=2, dash=(5, 2))
        # ttk.Label(curve_legend_content, text="检测阈值", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 坐标轴说明
        axis_label = ttk.Label(curve_legend_content, text="X:灰度值(0-255) | Y:像素数", font=('Arial', 8), foreground='gray')
        axis_label.pack(side=tk.LEFT, padx=(15, 0))

  
    def create_roi2_match_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ROI2命中对比")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        control_frame = ttk.Frame(frame, padding=(10, 10, 10, 6))
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        control_frame.columnconfigure(0, weight=0)
        control_frame.columnconfigure(1, weight=0)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=0)

        source_group = ttk.LabelFrame(control_frame, text="图片来源", padding=(10, 8))
        source_group.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))

        ttk.Button(
            source_group,
            text="导入图片",
            command=self.import_roi1_image
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            source_group,
            text="视频抽帧ROI1",
            command=self.extract_roi1_frames_from_video
        ).pack(side=tk.LEFT)

        scan_group = ttk.LabelFrame(control_frame, text="命中扫描", padding=(10, 8))
        scan_group.grid(row=0, column=1, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))

        scan_row = ttk.Frame(scan_group)
        scan_row.pack(anchor=tk.W)
        ttk.Label(scan_row, text="ROI3阈值").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(
            scan_row,
            textvariable=self.roi3_scan_threshold_var,
            width=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.scan_roi2_high_gray_button = ttk.Button(
            scan_row,
            textvariable=self.roi3_scan_button_text_var,
            command=self.scan_roi2_high_gray_matches
        )
        self.scan_roi2_high_gray_button.pack(side=tk.LEFT)

        self.load_offline_db_button = ttk.Button(
            scan_row,
            text="加载OFFLINE库图",
            command=self.load_latest_offline_db_images
        )
        self.load_offline_db_button.pack(side=tk.LEFT, padx=(8, 0))

        nav_group = ttk.LabelFrame(control_frame, text="命中导航", padding=(10, 8))
        nav_group.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        nav_group.columnconfigure(0, weight=1)

        nav_top_row = ttk.Frame(nav_group)
        nav_top_row.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.prev_roi2_high_gray_button = ttk.Button(
            nav_top_row,
            text="上一个命中",
            command=self.goto_previous_roi2_high_gray_match
        )
        self.prev_roi2_high_gray_button.pack(side=tk.LEFT, padx=(0, 8))

        self.next_roi2_high_gray_button = ttk.Button(
            nav_top_row,
            text="下一个命中",
            command=self.goto_next_roi2_high_gray_match
        )
        self.next_roi2_high_gray_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            nav_top_row,
            textvariable=self.roi2_high_gray_match_var,
            font=('Arial', 10, 'bold'),
            foreground='darkred'
        ).pack(side=tk.LEFT)

        nav_bottom_row = ttk.Frame(nav_group)
        nav_bottom_row.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(8, 0))

        self.toggle_roi2_neighbor_display_button = ttk.Button(
            nav_bottom_row,
            textvariable=self.roi2_neighbor_display_mode_var,
            command=self.toggle_roi2_neighbor_display_mode
        )
        self.toggle_roi2_neighbor_display_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            nav_bottom_row,
            textvariable=self.roi2_compare_threshold_hint_var,
            font=('Arial', 10),
            foreground='gray35'
        ).pack(side=tk.LEFT)

        hem_group = ttk.LabelFrame(control_frame, text="HEM判定", padding=(10, 8))
        hem_group.grid(row=0, column=3, sticky=(tk.E, tk.N, tk.S))

        hem_top_row = ttk.Frame(hem_group)
        hem_top_row.pack(anchor=tk.E, fill=tk.X)
        ttk.Label(hem_top_row, text="灰度差X").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(
            hem_top_row,
            textvariable=self.roi2_hem_avg_delta_var,
            width=6
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(hem_top_row, text="像素差X2").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(
            hem_top_row,
            textvariable=self.roi2_hem_pixel_delta_var,
            width=6
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.roi2_hem_result_label = tk.Label(
            hem_top_row,
            textvariable=self.roi2_hem_result_var,
            font=('Arial', 16, 'bold'),
            fg='gray45',
            bg=self.root.cget('bg'),
            padx=4
        )
        self.roi2_hem_result_label.pack(side=tk.LEFT)

        ttk.Label(
            frame,
            textvariable=self.roi2_compare_status_var,
            foreground='gray'
        ).grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 8))

        panels_frame = ttk.Frame(frame, padding=(10, 0, 10, 10))
        panels_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        panels_frame.rowconfigure(0, weight=1)

        self.roi2_compare_panels = []
        panel_specs = [
            ("previous", "上一张", False),
            ("next", "下一张", False),
            ("current", "当前命中", True),
        ]

        for column, (panel_role, title_prefix, is_current_panel) in enumerate(panel_specs):
            panels_frame.columnconfigure(column, weight=1)

            panel_title = self.get_roi2_compare_panel_title(
                title_prefix,
                is_current_panel,
                panel_role=panel_role
            )
            panel_frame = ttk.LabelFrame(panels_frame, text=panel_title, padding=10)
            panel_frame.grid(row=0, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
            panel_frame.columnconfigure(0, weight=1)
            panel_frame.rowconfigure(0, weight=1)

            panel_background = '#fff7e6' if is_current_panel else 'white'
            panel_border = '#d35400' if is_current_panel else '#b8b8b8'

            image_container = tk.Frame(
                panel_frame,
                bg=panel_background,
                bd=1,
                relief=tk.SOLID,
                highlightthickness=2,
                highlightbackground=panel_border
            )
            image_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            image_label = tk.Label(
                image_container,
                bg=panel_background,
                fg='gray40',
                text="无图片",
                font=('Arial', 11),
                width=26,
                height=12,
                wraplength=260,
                justify=tk.CENTER
            )
            image_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            image_container.bind("<Configure>", lambda event, idx=column: self.on_roi2_compare_panel_resize(idx, event))
            image_container.bind("<MouseWheel>", self.on_roi2_compare_mousewheel)
            image_container.bind("<Button-4>", self.on_roi2_compare_mousewheel)
            image_container.bind("<Button-5>", self.on_roi2_compare_mousewheel)
            image_label.bind("<MouseWheel>", self.on_roi2_compare_mousewheel)
            image_label.bind("<Button-4>", self.on_roi2_compare_mousewheel)
            image_label.bind("<Button-5>", self.on_roi2_compare_mousewheel)

            filename_var = tk.StringVar(value="无图片")
            avg_var = tk.StringVar(value="平均灰度: --")
            count_var = tk.StringVar(value=">阈值像素: --")

            ttk.Label(
                panel_frame,
                textvariable=filename_var,
                font=('Arial', 10, 'bold'),
                anchor=tk.CENTER
            ).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(8, 4))
            ttk.Label(
                panel_frame,
                textvariable=avg_var,
                foreground='green4'
            ).grid(row=2, column=0, sticky=tk.W, pady=2)
            ttk.Label(
                panel_frame,
                textvariable=count_var,
                foreground='green4'
            ).grid(row=3, column=0, sticky=tk.W, pady=2)

            self.roi2_compare_panels.append({
                'panel_frame': panel_frame,
                'panel_role': panel_role,
                'title_prefix': title_prefix,
                'image_label': image_label,
                'image_container': image_container,
                'filename_var': filename_var,
                'avg_var': avg_var,
                'count_var': count_var,
                'is_current_panel': is_current_panel,
                'show_full_image': is_current_panel,
                'background': panel_background,
                'last_size': (0, 0),
                'cached_file_path': None,
                'cached_payload': None,
                'cached_empty_message': "无图片",
                'cached_file_text': None,
            })

        self.update_roi2_high_gray_match_ui()
        self.update_roi2_neighbor_display_button_text()
        self.update_roi2_compare_panel_titles()
        self.update_roi2_compare_threshold_hint()
        self.refresh_roi2_high_gray_comparison()

    def create_peak_detection_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="峰值检测")

        self.peak_vars = {}

        configs = [
            ("peak_detection.threshold", "固定阈值", "float"),
            ("peak_detection.adaptive_threshold_enabled", "启用自适应阈值", "bool"),
            ("peak_detection.threshold_over_mean_ratio", "自适应阈值上浮比例", "float"),
            ("peak_detection.difference_threshold", "绿红分类阈值", "float"),
            ("peak_detection.margin_frames", "峰间最小间隔(帧)", "int"),
            ("peak_detection.silence_frames", "干净区间长度(帧)", "int"),
            ("peak_detection.min_region_length", "最小波峰宽度(帧)", "int"),
            ("peak_detection.pre_post_avg_frames", "平均值窗口帧数", "int"),
            ("peak_detection.adaptive_window_seconds", "自适应时间窗口(秒)", "float"),
        ]

        row = 0
        for key, label, config_type in configs:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
            else:
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)

            self.peak_vars[key] = var
            row += 1

        # 添加说明
        help_text = """说明:
- 固定阈值: 基础的灰度阈值
- 自适应阈值: 阈值=最近N秒平均值*(1+上浮比例)
- 绿红分类阈值: 峰后平均值 - 峰前平均值 > 阈值则为绿色，否则红色
- 峰间最小间隔: 两个波峰间隔小于此值只保留峰值更高的
- 干净区间长度: 波峰前后必须连续低于阈值的帧数"""

        help_label = ttk.Label(frame, text=help_text, justify=tk.LEFT,
                               font=('Arial', 9), foreground='gray')
        help_label.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

    def create_roi3_override_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ROI3覆盖")

        self.roi3_override_vars = {}

        # 主配置
        ttk.Label(frame, text="ROI3覆盖逻辑配置", font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=3, pady=(10, 15))

        # 启用开关
        row = 1
        ttk.Label(frame, text="启用ROI3覆盖功能:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)
        enabled_var = tk.BooleanVar()
        ttk.Checkbutton(frame, variable=enabled_var,
                       command=self.on_roi3_override_toggle).grid(row=row, column=1, sticky=tk.W, padx=5, pady=8)
        self.roi3_override_vars["peak_detection.roi3_override.enabled"] = enabled_var
        row += 1

        # 分隔线
        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                      sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 阈值设置
        ttk.Label(frame, text="ROI3峰值阈值:", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        threshold_var = tk.StringVar()
        threshold_entry = ttk.Entry(frame, textvariable=threshold_var, width=15)
        threshold_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        self.roi3_override_vars["peak_detection.roi3_override.threshold"] = threshold_var

        # 添加阈值说明
        ttk.Label(frame, text="(当ROI3峰值大于此值时，红色波峰将被覆盖为绿色)",
                 font=('Arial', 9), foreground='blue').grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        # 要求ROI3数据
        ttk.Label(frame, text="要求有效的ROI3数据:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)
        require_var = tk.BooleanVar()
        ttk.Checkbutton(frame, variable=require_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=8)
        self.roi3_override_vars["peak_detection.roi3_override.require_roi3_data"] = require_var
        row += 1

        # 预设值按钮
        ttk.Label(frame, text="快速设置:", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=10)
        row += 1

        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(preset_frame, text="保守 (100)",
                  command=lambda: self.set_roi3_threshold(100)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="默认 (115)",
                  command=lambda: self.set_roi3_threshold(115)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="积极 (130)",
                  command=lambda: self.set_roi3_threshold(130)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="极高 (150)",
                  command=lambda: self.set_roi3_threshold(150)).pack(side=tk.LEFT, padx=3)
        row += 1

        # 分隔线
        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                      sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 实时预览区域
        ttk.Label(frame, text="配置预览", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, pady=(5, 10))
        row += 1

        preview_frame = ttk.LabelFrame(frame, text="当前配置", padding=10)
        preview_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.preview_text = tk.Text(preview_frame, height=6, width=70, wrap=tk.WORD,
                                   font=('Consolas', 10))
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        # 更新预览
        threshold_var.trace('w', self.update_roi3_preview)
        enabled_var.trace('w', self.update_roi3_preview)
        self.update_roi3_preview()

        # 使用说明
        row += 1
        help_text = """ROI3覆盖逻辑说明:
1. 当ROI2检测为红色波峰时，系统会检查ROI3的峰值
2. 如果ROI3峰值 > 设定阈值，则将波峰覆盖为绿色
3. 这允许使用ROI3区域的高回声信号来"纠正"ROI2的分类
4. 适用于ROI2区域信号不稳定但ROI3区域信号更可靠的情况

建议设置:
- 100: 保守设置，更容易触发覆盖
- 115: 默认设置，平衡敏感性和特异性
- 130: 积极设置，需要更强的ROI3信号
- 150: 极高设置，只在强信号时触发"""

        help_frame = ttk.LabelFrame(frame, text="使用说明", padding=10)
        help_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=10)

        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT,
                              font=('Arial', 9))
        help_label.pack(fill=tk.BOTH, expand=True)

        frame.columnconfigure(1, weight=1)

    def set_roi3_threshold(self, value):
        """设置ROI3阈值"""
        self.roi3_override_vars["roi3_override.threshold"].set(str(value))

    def update_roi3_preview(self, *args):
        """更新ROI3配置预览"""
        try:
            enabled = self.roi3_override_vars["roi3_override.enabled"].get()
            threshold = self.roi3_override_vars["roi3_override.threshold"].get()
            require_data = self.roi3_override_vars["roi3_override.require_roi3_data"].get()

            if not enabled:
                preview = "ROI3覆盖功能: 已禁用\n\n系统将仅使用ROI2进行波峰分类，不会应用ROI3覆盖逻辑。"
            else:
                try:
                    threshold_val = float(threshold)
                    preview = f"""ROI3覆盖功能: 已启用

阈值设置: {threshold_val}
要求ROI3数据: {'是' if require_data else '否'}

工作逻辑:
- 当ROI2检测为红色波峰时
- 检查ROI3峰值是否 > {threshold_val}
- 如果是: 红色 → 绿色 (覆盖)
- 如果否: 保持红色不变"""
                except ValueError:
                    preview = "错误: 阈值设置无效，请输入有效数字"

            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview)

        except Exception as e:
            pass

    def on_roi3_override_toggle(self):
        """ROI3覆盖开关切换时的处理"""
        enabled = self.roi3_override_vars["roi3_override.enabled"].get()
        # 可以在这里添加启用/禁用时需要执行的其他操作

    def browse_file(self, var):
        """浏览文件"""
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            var.set(filename)

    def load_config(self):
        """加载配置文件"""
        try:
            self.ensure_runtime_config_file()
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            else:
                self.config_data = {}
                print(f"[WARNING] 配置文件 {self.config_path} 不存在，将创建新配置")

            # 将配置数据加载到UI控件
            self.load_to_widgets()

            self.status_var.set(f"配置已加载: {self.config_path}")

        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("加载失败")

    def ensure_runtime_config_file(self):
        """打包运行时，首次启动时将默认配置复制到 exe 同目录。"""
        if os.path.exists(self.config_path):
            return

        if not os.path.exists(self.bundled_config_path):
            return

        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(Path(self.bundled_config_path).read_bytes())
        print(f"[INFO] 已初始化运行时配置文件: {self.config_path}")

    def load_to_widgets(self):
        """将配置数据加载到UI控件"""
        # 基础配置
        for key, var in self.basic_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # ROI配置
        for key, var in self.roi_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # 峰值检测配置
        for key, var in self.peak_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # ROI3覆盖配置
        for key, var in self.roi3_override_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # 确保 ROI 默认参数有默认值，避免首次运行时关键按钮无响应
        roi_defaults = {
            "roi_capture.default_config.x1": "1280",
            "roi_capture.default_config.y1": "80",
            "roi_capture.default_config.x2": "1920",
            "roi_capture.default_config.y2": "980",
            "roi_capture.roi2_config.extension_params.left": "40",
            "roi_capture.roi2_config.extension_params.right": "40",
            "roi_capture.roi2_config.extension_params.top": "50",
            "roi_capture.roi2_config.extension_params.bottom": "30",
        }

        for key, default_value in roi_defaults.items():
            if key in self.roi_vars and (not self.roi_vars[key].get() or self.roi_vars[key].get().strip() == ""):
                self.roi_vars[key].set(default_value)

        # 确保ROI3扩展参数有默认值
        roi3_defaults = {
            "roi_capture.roi3_config.extension_params.left": "30",
            "roi_capture.roi3_config.extension_params.right": "40",
            "roi_capture.roi3_config.extension_params.top": "70",
            "roi_capture.roi3_config.extension_params.bottom": "30"
        }

        for key, default_value in roi3_defaults.items():
            if key in self.roi_vars and (not self.roi_vars[key].get() or self.roi_vars[key].get().strip() == ""):
                self.roi_vars[key].set(default_value)

    def get_ui_state_var_groups(self):
        """获取需要自动持久化的UI变量分组"""
        return {
            "basic_vars": self.basic_vars,
            "roi_vars": self.roi_vars,
            "peak_vars": self.peak_vars,
            "roi3_override_vars": self.roi3_override_vars,
            "ui_runtime_vars": {
                "threshold_lower": self.threshold_lower_var,
                "threshold_upper": self.threshold_upper_var,
                "roi2_stats_threshold": self.roi2_stats_threshold_var,
                "roi3_scan_threshold": self.roi3_scan_threshold_var,
                "roi2_neighbor_show_full_image": self.roi2_neighbor_show_full_image_var,
                "roi2_hem_avg_delta": self.roi2_hem_avg_delta_var,
                "roi2_hem_pixel_delta": self.roi2_hem_pixel_delta_var,
                "continuous_check_enabled": self.continuous_check_enabled,
                "overlay_enabled": self.overlay_enabled,
                "overlay_alpha": self.overlay_alpha,
                "continuous_heatmap_enabled": self.continuous_heatmap_enabled,
                "heatmap_alpha": self.heatmap_alpha_var,
            }
        }

    def register_ui_state_watchers(self):
        """注册UI状态自动保存监听"""
        if self.ui_state_watchers_registered:
            return

        for var_group in self.get_ui_state_var_groups().values():
            for var in var_group.values():
                var.trace_add('write', lambda *args: self.schedule_ui_state_save())

        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.schedule_ui_state_save(), add="+")
        self.ui_state_watchers_registered = True

    def collect_ui_state(self):
        """收集当前UI状态"""
        widget_values = {}
        for group_name, var_group in self.get_ui_state_var_groups().items():
            widget_values[group_name] = {}
            for key, var in var_group.items():
                widget_values[group_name][key] = var.get()

        return {
            "version": 1,
            "widget_values": widget_values,
            "selected_tab_index": self.notebook.index(self.notebook.select())
        }

    def apply_ui_state_value(self, var, value):
        """将保存的值恢复到Tk变量"""
        try:
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
                var.set(value)
            else:
                var.set("" if value is None else str(value))
        except tk.TclError as e:
            print(f"[WARNING] UI状态恢复失败，值无效: {value} ({e})")

    def load_ui_state(self):
        """加载自动保存的UI状态"""
        try:
            if not os.path.exists(self.ui_state_path):
                return

            with open(self.ui_state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            widget_values = state_data.get("widget_values", {})
            for group_name, var_group in self.get_ui_state_var_groups().items():
                saved_group = widget_values.get(group_name, {})
                for key, var in var_group.items():
                    if key in saved_group:
                        self.apply_ui_state_value(var, saved_group[key])

            selected_tab_index = state_data.get("selected_tab_index")
            if isinstance(selected_tab_index, int) and 0 <= selected_tab_index < len(self.notebook.tabs()):
                self.notebook.select(selected_tab_index)

            self.alpha_value_label.config(text=f"{self.overlay_alpha.get():.2f}")
            print(f"[INFO] 已恢复UI状态: {self.ui_state_path}")

        except Exception as e:
            print(f"[ERROR] 加载UI状态失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_ui_state(self):
        """保存当前UI状态到独立文件"""
        try:
            state_data = self.collect_ui_state()
            with open(self.ui_state_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            self.ui_state_save_job = None
            print(f"[INFO] UI状态已保存: {self.ui_state_path}")
            return True
        except Exception as e:
            self.ui_state_save_job = None
            print(f"[ERROR] 保存UI状态失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def schedule_ui_state_save(self):
        """延迟保存UI状态，避免频繁写盘"""
        if self.ui_state_save_job is not None:
            self.root.after_cancel(self.ui_state_save_job)

        self.ui_state_save_job = self.root.after(400, self.save_ui_state)

    def on_window_close(self):
        self.cancel_roi2_high_gray_scan_poll()
        if self.roi2_compare_resize_job is not None:
            self.root.after_cancel(self.roi2_compare_resize_job)
            self.roi2_compare_resize_job = None
        if self.ui_state_save_job is not None:
            self.root.after_cancel(self.ui_state_save_job)
            self.ui_state_save_job = None
        self.save_ui_state()
        self.root.destroy()

    def get_nested_value(self, key_path: str, data: dict):
        """获取嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def set_nested_value(self, key_path: str, data: dict, value):
        """设置嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def save_config(self):
        """保存配置到文件"""
        try:
            # 从UI控件收集配置数据
            self.collect_from_widgets()

            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)

            self.save_ui_state()
            self.status_var.set(f"配置已保存: {self.config_path}")
            print("[INFO] 配置已成功保存!")

        except Exception as e:
            print(f"[ERROR] 保存配置文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("保存失败")

    def save_as(self):
        """另存为配置文件"""
        try:
            filename = filedialog.asksaveasfilename(
                title="另存为配置文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                # 从UI控件收集配置数据
                self.collect_from_widgets()

                # 保存到指定文件
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)

                self.save_ui_state()
                self.status_var.set(f"配置已保存: {filename}")
                print(f"[INFO] 配置已保存到: {filename}")

        except Exception as e:
            print(f"[ERROR] 另存为失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("另存为失败")

    def collect_from_widgets(self):
        """从UI控件收集配置数据"""
        # 基础配置
        for key, var in self.basic_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                # 尝试转换为适当的类型
                if value:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # ROI配置
        for key, var in self.roi_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # 峰值检测配置
        for key, var in self.peak_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # ROI3覆盖配置
        for key, var in self.roi3_override_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

    def update_roi2_high_gray_match_ui(self):
        """更新ROI2高灰度命中按钮和计数显示"""
        total_matches = len(self.roi2_high_gray_match_paths)
        total_offline_records = len(self.offline_db_records)
        if self.roi2_compare_mode == "offline_db":
            if total_offline_records <= 0:
                self.roi2_high_gray_match_var.set("库图: 0/0")
            elif 0 <= self.current_offline_db_position < total_offline_records:
                self.roi2_high_gray_match_var.set(
                    f"库图: {self.current_offline_db_position + 1}/{total_offline_records}"
                )
            else:
                self.roi2_high_gray_match_var.set(f"库图: 0/{total_offline_records}")
        elif self.is_scanning_roi2_high_gray:
            self.roi2_high_gray_match_var.set("命中: 扫描中...")
        elif total_matches <= 0:
            self.roi2_high_gray_match_var.set("命中: 0/0")
        elif 0 <= self.current_roi2_match_position < total_matches:
            self.roi2_high_gray_match_var.set(f"命中: {self.current_roi2_match_position + 1}/{total_matches}")
        else:
            self.roi2_high_gray_match_var.set(f"命中: 0/{total_matches}")

        scan_button_state = (
            tk.NORMAL
            if self.current_roi1_path and not self.is_scanning_roi2_high_gray and self.get_roi3_scan_threshold() is not None
            else tk.DISABLED
        )
        load_offline_button_state = tk.DISABLED if self.is_scanning_roi2_high_gray else tk.NORMAL
        nav_button_state = tk.NORMAL if total_matches > 0 and not self.is_scanning_roi2_high_gray else tk.DISABLED
        if self.roi2_compare_mode == "offline_db":
            nav_button_state = tk.NORMAL if total_offline_records > 0 and not self.is_scanning_roi2_high_gray else tk.DISABLED

        if self.scan_roi2_high_gray_button is not None:
            self.scan_roi2_high_gray_button.config(state=scan_button_state)
        if self.load_offline_db_button is not None:
            self.load_offline_db_button.config(state=load_offline_button_state)
        if self.prev_roi2_high_gray_button is not None:
            self.prev_roi2_high_gray_button.config(state=nav_button_state)
        if self.next_roi2_high_gray_button is not None:
            self.next_roi2_high_gray_button.config(state=nav_button_state)

    def get_roi3_scan_threshold(self):
        """获取 ROI3 命中扫描使用的平均灰度阈值。"""
        try:
            raw_value = self.roi3_scan_threshold_var.get().strip()
            if not raw_value:
                return None

            threshold = int(raw_value)
            if 0 <= threshold <= 255:
                return threshold

            return None
        except (ValueError, tk.TclError, AttributeError):
            return None

    def get_roi3_scan_target_text(self):
        threshold = self.get_roi3_scan_threshold()
        if threshold is None:
            return "ROI3>无效"
        return f"ROI3>{threshold}"

    def update_roi3_scan_button_text(self):
        self.roi3_scan_button_text_var.set(f"扫描{self.get_roi3_scan_target_text()}")

    def on_roi3_scan_threshold_changed(self):
        self.update_roi3_scan_button_text()

        if self.roi2_high_gray_match_paths or self.is_scanning_roi2_high_gray:
            self.invalidate_roi2_high_gray_matches(
                f"ROI3扫描阈值已变化，请重新扫描 {self.get_roi3_scan_target_text()} 命中结果"
            )
            return

        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()

    def on_roi2_stats_threshold_changed(self):
        self.update_roi_visualization()
        self.update_roi2_compare_threshold_hint()
        self.refresh_roi2_high_gray_comparison()

    def update_roi2_compare_threshold_hint(self):
        threshold = self.get_roi2_stats_threshold()
        if threshold is None:
            self.roi2_compare_threshold_hint_var.set("计数阈值: 无效")
        else:
            self.roi2_compare_threshold_hint_var.set(f"计数阈值: {threshold}")

    def parse_float_threshold_var(self, threshold_var):
        try:
            raw_value = threshold_var.get().strip()
            if not raw_value:
                return None
            return float(raw_value)
        except (ValueError, tk.TclError, AttributeError):
            return None

    def get_roi2_hem_thresholds(self):
        return (
            self.parse_float_threshold_var(self.roi2_hem_avg_delta_var),
            self.parse_float_threshold_var(self.roi2_hem_pixel_delta_var),
        )

    def set_roi2_hem_result(self, text, color):
        self.roi2_hem_result_var.set(text)
        if self.roi2_hem_result_label is not None:
            self.roi2_hem_result_label.configure(fg=color)

    def get_cached_roi2_compare_payload_by_role(self, role):
        for panel in self.roi2_compare_panels:
            if panel.get('panel_role') == role:
                return panel.get('cached_payload')
        return None

    def refresh_roi2_compare_judgement(self):
        avg_threshold, pixel_threshold = self.get_roi2_hem_thresholds()
        previous_payload = self.get_cached_roi2_compare_payload_by_role('previous')
        next_payload = self.get_cached_roi2_compare_payload_by_role('next')

        if avg_threshold is None and pixel_threshold is None:
            self.set_roi2_hem_result("--", "gray45")
            return

        if previous_payload is None or next_payload is None:
            self.set_roi2_hem_result("Fail", "red3")
            return

        previous_stats = previous_payload.get('stats')
        next_stats = next_payload.get('stats')
        if previous_stats is None or next_stats is None:
            self.set_roi2_hem_result("Fail", "red3")
            return

        avg_delta = next_stats['avg_gray'] - previous_stats['avg_gray']
        avg_match = avg_threshold is not None and avg_delta > avg_threshold

        prev_pixels = previous_stats.get('pixels_above_threshold')
        next_pixels = next_stats.get('pixels_above_threshold')
        pixel_match = (
            pixel_threshold is not None and
            prev_pixels is not None and
            next_pixels is not None and
            (next_pixels - prev_pixels) > pixel_threshold
        )

        if avg_match or pixel_match:
            self.set_roi2_hem_result("HEM", "green4")
        else:
            self.set_roi2_hem_result("Fail", "red3")

    def on_roi2_hem_threshold_changed(self):
        self.refresh_roi2_compare_judgement()

    def get_roi2_compare_panel_title(self, title_prefix, is_current_panel, panel_role=None):
        if self.roi2_compare_mode == "offline_db":
            title_map = {
                'previous': "Before",
                'next': "After",
                'current': "Differ",
            }
            return title_map.get(panel_role, title_prefix)

        if is_current_panel:
            return f"{title_prefix}整图"
        if self.roi2_neighbor_show_full_image_var.get():
            return f"{title_prefix}整图"
        return f"{title_prefix} ROI2"

    def update_roi2_neighbor_display_button_text(self):
        if self.roi2_neighbor_show_full_image_var.get():
            self.roi2_neighbor_display_mode_var.set("上/下张显示: 整图")
        else:
            self.roi2_neighbor_display_mode_var.set("上/下张显示: ROI2")

    def update_roi2_compare_panel_titles(self):
        for panel in self.roi2_compare_panels:
            is_current_panel = panel.get('is_current_panel', False)
            title_prefix = panel.get('title_prefix', "")
            panel_role = panel.get('panel_role')
            if self.roi2_compare_mode == "offline_db":
                panel['show_full_image'] = True
            else:
                panel['show_full_image'] = is_current_panel or self.roi2_neighbor_show_full_image_var.get()
            panel_frame = panel.get('panel_frame')
            if panel_frame is not None:
                panel_frame.configure(
                    text=self.get_roi2_compare_panel_title(
                        title_prefix,
                        is_current_panel,
                        panel_role=panel_role
                    )
                )

    def toggle_roi2_neighbor_display_mode(self):
        self.roi2_neighbor_show_full_image_var.set(not self.roi2_neighbor_show_full_image_var.get())

    def on_roi2_neighbor_display_mode_changed(self, *args):
        self.update_roi2_neighbor_display_button_text()
        if not self.roi2_compare_panels:
            return
        self.update_roi2_compare_panel_titles()
        self.refresh_roi2_high_gray_comparison()

    def clear_roi2_compare_panel(self, panel, image_text="无图片", file_text=None, file_path=None):
        panel['filename_var'].set(file_text or image_text)
        panel['avg_var'].set("平均灰度: --")
        panel['count_var'].set(">阈值像素: --")
        panel['image_label'].configure(
            image='',
            text=image_text,
            bg=panel['background']
        )
        panel['image_label'].image = None
        panel['cached_file_path'] = file_path
        panel['cached_payload'] = None
        panel['cached_empty_message'] = image_text
        panel['cached_file_text'] = file_text

    def get_mousewheel_delta(self, event):
        if getattr(event, 'delta', 0):
            return 1 if event.delta > 0 else -1
        if getattr(event, 'num', None) == 4:
            return 1
        if getattr(event, 'num', None) == 5:
            return -1
        return 0

    def has_cached_roi2_compare_content(self):
        return any(panel.get('cached_payload') is not None for panel in self.roi2_compare_panels)

    def rerender_roi2_compare_panels_from_cache(self):
        if not self.roi2_compare_panels:
            return False

        threshold = self.get_roi2_stats_threshold()
        had_cached_content = False

        for panel in self.roi2_compare_panels:
            file_path = panel.get('cached_file_path')
            payload = panel.get('cached_payload')
            empty_message = panel.get('cached_empty_message', "无图片")
            file_text = panel.get('cached_file_text')

            if file_path and payload is not None:
                had_cached_content = True
                self.update_roi2_compare_panel(panel, file_path, payload, threshold, empty_message)
                continue

            self.clear_roi2_compare_panel(
                panel,
                empty_message,
                file_text=file_text,
                file_path=file_path
            )

        return had_cached_content

    def on_roi2_compare_mousewheel(self, event):
        delta = self.get_mousewheel_delta(event)
        if delta == 0 or not self.has_cached_roi2_compare_content():
            return "break"

        current_zoom = self.roi2_compare_zoom_factor
        if delta > 0:
            new_zoom = current_zoom * (1 + self.roi2_compare_zoom_step)
        else:
            new_zoom = current_zoom * (1 - self.roi2_compare_zoom_step)

        new_zoom = max(self.roi2_compare_min_zoom, min(self.roi2_compare_max_zoom, new_zoom))
        if abs(new_zoom - current_zoom) <= 0.001:
            return "break"

        self.roi2_compare_zoom_factor = new_zoom
        zoom_percentage = int(round(new_zoom * 100))
        self.status_var.set(f"ROI2命中对比缩放: {zoom_percentage}%")
        self.rerender_roi2_compare_panels_from_cache()
        return "break"

    def on_roi2_compare_panel_resize(self, panel_index, event):
        if panel_index < 0 or panel_index >= len(self.roi2_compare_panels):
            return

        panel = self.roi2_compare_panels[panel_index]
        new_size = (event.width, event.height)
        if panel.get('last_size') == new_size:
            return

        panel['last_size'] = new_size
        self.schedule_roi2_compare_refresh()

    def schedule_roi2_compare_refresh(self):
        if self.roi2_compare_resize_job is not None:
            try:
                self.root.after_cancel(self.roi2_compare_resize_job)
            except tk.TclError:
                pass

        self.roi2_compare_resize_job = self.root.after(80, self.refresh_roi2_high_gray_comparison_from_resize)

    def refresh_roi2_high_gray_comparison_from_resize(self):
        self.roi2_compare_resize_job = None
        if not self.rerender_roi2_compare_panels_from_cache():
            self.refresh_roi2_high_gray_comparison()

    def get_roi2_compare_display_size(self, panel, default_width=260, default_height=220):
        container = panel.get('image_container')
        width = container.winfo_width() if container is not None else 0
        height = container.winfo_height() if container is not None else 0

        usable_width = max(1, width - 12)
        usable_height = max(1, height - 12)

        if width <= 1 or height <= 1:
            return default_width, default_height

        return usable_width, usable_height

    def add_roi2_outline_to_full_image(self, image, roi2_box):
        if image is None:
            return None

        display_image = image.copy()
        if display_image.mode != 'RGB':
            display_image = display_image.convert('RGB')

        if not roi2_box:
            return display_image

        x1, y1, x2, y2 = roi2_box
        if x2 <= x1 or y2 <= y1:
            return display_image

        outline_width = max(3, int(round(min(display_image.size) * 0.006)))
        draw = ImageDraw.Draw(display_image)
        draw.rectangle((x1, y1, x2, y2), outline='red', width=outline_width)
        return display_image

    def get_roi2_compare_display_image(self, panel, roi2_payload):
        if roi2_payload is None:
            return None

        if panel.get('show_full_image'):
            return self.add_roi2_outline_to_full_image(
                roi2_payload.get('full_image'),
                roi2_payload.get('roi2_box_in_full_image')
            )

        return roi2_payload.get('roi2_image')

    def create_roi2_compare_photo(self, image, panel=None, max_width=260, max_height=220):
        if image is None:
            return None

        display_image = image.copy()
        if display_image.mode != 'RGB':
            display_image = display_image.convert('RGB')

        if panel is not None:
            max_width, max_height = self.get_roi2_compare_display_size(
                panel,
                default_width=max_width,
                default_height=max_height
            )

        width, height = display_image.size
        if width <= 0 or height <= 0:
            return None

        scale = min(max_width / width, max_height / height) * self.roi2_compare_zoom_factor
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))
        resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS

        if display_width != width or display_height != height:
            display_image = display_image.resize((display_width, display_height), resample)

        return ImageTk.PhotoImage(display_image)

    def update_roi2_compare_panel(self, panel, file_path, roi2_payload, threshold, empty_message):
        if not file_path:
            self.clear_roi2_compare_panel(panel, empty_message)
            return

        file_name = os.path.basename(file_path)
        if roi2_payload is None:
            self.clear_roi2_compare_panel(panel, "ROI2加载失败", file_name, file_path=file_path)
            return

        stats = roi2_payload.get('stats')
        panel['filename_var'].set(file_name)
        panel['cached_file_path'] = file_path
        panel['cached_payload'] = roi2_payload
        panel['cached_empty_message'] = empty_message
        panel['cached_file_text'] = file_name

        if stats is None:
            self.clear_roi2_compare_panel(panel, "ROI2区域无效", file_name, file_path=file_path)
            if threshold is None:
                panel['count_var'].set(">阈值像素: 阈值无效")
            else:
                panel['count_var'].set(f">{threshold} 像素: --")
            return

        panel['avg_var'].set(f"平均灰度: {stats['avg_gray']:.2f}")
        if threshold is None:
            panel['count_var'].set(">阈值像素: 阈值无效")
        else:
            panel['count_var'].set(f">{threshold} 像素: {stats['pixels_above_threshold']}")

        display_image = self.get_roi2_compare_display_image(panel, roi2_payload)
        photo = self.create_roi2_compare_photo(display_image, panel=panel)
        if photo is None:
            fallback_message = "原图加载失败" if panel.get('show_full_image') else "ROI2区域无效"
            self.clear_roi2_compare_panel(panel, fallback_message, file_name, file_path=file_path)
            if threshold is None:
                panel['count_var'].set(">阈值像素: 阈值无效")
            else:
                panel['count_var'].set(f">{threshold} 像素: {stats['pixels_above_threshold']}")
            panel['avg_var'].set(f"平均灰度: {stats['avg_gray']:.2f}")
            return

        panel['image_label'].configure(
            image=photo,
            text='',
            bg=panel['background']
        )
        panel['image_label'].image = photo

    def get_current_roi2_compare_paths(self):
        if not self.current_roi1_path or not self.folder_image_files:
            return None

        current_folder_index = self.find_folder_image_index(self.current_roi1_path)
        if current_folder_index < 0:
            return None

        previous_path = self.folder_image_files[current_folder_index - 1] if current_folder_index > 0 else None
        current_path = self.folder_image_files[current_folder_index]
        next_path = (
            self.folder_image_files[current_folder_index + 1]
            if current_folder_index < len(self.folder_image_files) - 1
            else None
        )
        return {
            'previous': previous_path,
            'current': current_path,
            'next': next_path,
        }

    def parse_offline_image_record(self, row):
        if row is None:
            raise ValueError("OFFLINE 数据库记录为空")

        if len(row) != 4:
            raise ValueError(f"OFFLINE 记录字段数量错误，期望4，实际{len(row)}")

        record_id, point_id, image_path_text, modify_time = row
        image_path_text = "" if image_path_text is None else str(image_path_text).strip()
        parts = [part.strip() for part in image_path_text.split(";")]
        if len(parts) != 3:
            raise ValueError(
                f"ImagePath格式错误，必须为before;after;differ三段，实际为{len(parts)}段: {image_path_text}"
            )
        if any(not part for part in parts):
            raise ValueError(f"ImagePath存在空路径: {image_path_text}")

        before_path, after_path, differ_path = parts
        missing_paths = [path for path in parts if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError("图片文件不存在: " + " | ".join(missing_paths))

        return {
            "id": record_id,
            "point_id": point_id,
            "modify_time": modify_time,
            "before_path": before_path,
            "after_path": after_path,
            "differ_path": differ_path,
        }

    def query_offline_image_records(self):
        db_path = self.offline_db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")

        sql = (
            "SELECT ID, PointID, ImagePath, ModifyTime "
            "FROM SegmentImagesInfo "
            "WHERE ImagePath IS NOT NULL AND ImagePath LIKE '%;%;%' "
            "ORDER BY ModifyTime DESC, ID DESC"
        )

        with sqlite3.connect(db_path, timeout=30) as db:
            rows = db.cursor().execute(sql).fetchall()

        return rows

    def query_latest_offline_image_record(self):
        rows = self.query_offline_image_records()
        if not rows:
            return None
        return self.parse_offline_image_record(rows[0])

    def apply_offline_db_record_position(self, position):
        total = len(self.offline_db_records)
        if total <= 0:
            self.current_offline_db_position = -1
            self.offline_db_record = None
            return None

        if not (0 <= position < total):
            raise ValueError(f"数据库记录索引越界: {position}, total={total}")

        row = self.offline_db_records[position]
        record = self.parse_offline_image_record(row)
        self.current_offline_db_position = position
        self.offline_db_record = record
        return record

    def load_latest_offline_db_images(self):
        try:
            rows = self.query_offline_image_records()
            if not rows:
                msg = f"数据库中无可用 OFFLINE 记录: {self.offline_db_path}"
                self.status_var.set(msg)
                messagebox.showwarning("未找到OFFLINE记录", msg)
                return

            self.offline_db_records = rows
            self.apply_offline_db_record_position(0)
        except Exception as e:
            self.offline_db_records = []
            self.current_offline_db_position = -1
            self.offline_db_record = None
            msg = f"加载 OFFLINE 库图失败: {e}"
            self.status_var.set(msg)
            messagebox.showerror("加载失败", msg)
            return

        self.roi2_compare_mode = "offline_db"
        self.update_roi2_compare_panel_titles()
        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()

    def refresh_offline_db_comparison(self):
        if not self.roi2_compare_panels:
            return

        if self.offline_db_record is None:
            self.roi2_compare_status_var.set("请点击“加载OFFLINE库图”")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, "无图片")
            self.refresh_roi2_compare_judgement()
            return

        compare_paths = {
            'previous': self.offline_db_record.get("before_path"),
            'next': self.offline_db_record.get("after_path"),
            'current': self.offline_db_record.get("differ_path"),
        }
        default_messages = {
            'previous': "无Before图片",
            'next': "无After图片",
            'current': "无Differ图片",
        }

        roi_config = self.get_roi_config_values()
        threshold = self.get_roi2_stats_threshold()

        for panel in self.roi2_compare_panels:
            panel_role = panel.get('panel_role')
            file_path = compare_paths.get(panel_role)
            empty_message = default_messages.get(panel_role, "无图片")
            if not file_path:
                self.clear_roi2_compare_panel(panel, empty_message)
                continue

            roi2_payload = self.load_roi2_image_payload_from_path(
                file_path,
                roi_config,
                threshold,
                include_full_image=True
            )
            self.update_roi2_compare_panel(panel, file_path, roi2_payload, threshold, empty_message)

        total = len(self.offline_db_records)
        if 0 <= self.current_offline_db_position < total:
            position_text = f"{self.current_offline_db_position + 1}/{total}"
        else:
            position_text = f"0/{total}"

        self.roi2_compare_status_var.set(
            f"OFFLINE库图: {position_text} | "
            f"ID={self.offline_db_record.get('id')} | "
            f"PointID={self.offline_db_record.get('point_id')} | "
            f"ModifyTime={self.offline_db_record.get('modify_time')}"
        )
        self.refresh_roi2_compare_judgement()

    def refresh_roi2_high_gray_comparison(self):
        self.update_roi2_compare_threshold_hint()
        if self.roi2_compare_mode == "offline_db":
            self.refresh_offline_db_comparison()
            return

        scan_target_text = self.get_roi3_scan_target_text()
        if not self.roi2_compare_panels:
            return

        default_messages = {
            'previous': "无上一张",
            'next': "无下一张",
            'current': "无当前图片",
        }

        if not self.current_roi1_path:
            self.roi2_compare_status_var.set(f"请先导入图片并扫描 {scan_target_text} 命中结果")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, default_messages.get(panel.get('panel_role'), "无图片"))
            self.refresh_roi2_compare_judgement()
            return

        if self.get_roi3_scan_threshold() is None:
            self.roi2_compare_status_var.set("请先设置有效的 ROI3 扫描阈值 (0-255)")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, default_messages.get(panel.get('panel_role'), "无图片"))
            self.refresh_roi2_compare_judgement()
            return

        if self.is_scanning_roi2_high_gray:
            self.roi2_compare_status_var.set(f"正在扫描 {scan_target_text} 命中结果...")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, "扫描中...")
            self.refresh_roi2_compare_judgement()
            return

        if not self.roi2_high_gray_match_paths:
            self.roi2_compare_status_var.set(f"请先扫描 {scan_target_text} 命中结果")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, default_messages.get(panel.get('panel_role'), "无图片"))
            self.refresh_roi2_compare_judgement()
            return

        if self.current_roi2_match_position < 0:
            self.roi2_compare_status_var.set("当前图片不是命中图，请使用上一个命中/下一个命中")
            inactive_messages = {
                'previous': "无上一张",
                'next': "无下一张",
                'current': "当前图片不是命中图",
            }
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, inactive_messages.get(panel.get('panel_role'), "无图片"))
            self.refresh_roi2_compare_judgement()
            return

        compare_paths = self.get_current_roi2_compare_paths()
        if compare_paths is None:
            self.roi2_compare_status_var.set("当前命中图不在已扫描文件列表中，请重新扫描")
            for panel in self.roi2_compare_panels:
                self.clear_roi2_compare_panel(panel, default_messages.get(panel.get('panel_role'), "无图片"))
            self.refresh_roi2_compare_judgement()
            return

        roi_config = self.get_roi_config_values()
        threshold = self.get_roi2_stats_threshold()

        for panel in self.roi2_compare_panels:
            panel_role = panel.get('panel_role')
            file_path = compare_paths.get(panel_role)
            empty_message = default_messages.get(panel_role, "无图片")
            if file_path is None:
                self.clear_roi2_compare_panel(panel, empty_message)
                continue

            roi2_payload = self.load_roi2_image_payload_from_path(
                file_path,
                roi_config,
                threshold,
                include_full_image=panel.get('show_full_image', False)
            )
            self.update_roi2_compare_panel(panel, file_path, roi2_payload, threshold, empty_message)

        compare_names = []
        for panel in self.roi2_compare_panels:
            panel_role = panel.get('panel_role')
            file_path = compare_paths.get(panel_role)
            empty_message = default_messages.get(panel_role, "无图片")
            compare_names.append(os.path.basename(file_path) if file_path else empty_message)

        self.roi2_compare_status_var.set(
            f"当前命中: {self.current_roi2_match_position + 1}/{len(self.roi2_high_gray_match_paths)} | "
            f"对比: {compare_names[0]} / {compare_names[1]} / {compare_names[2]}"
        )
        self.refresh_roi2_compare_judgement()

    def cancel_roi2_high_gray_scan_poll(self):
        if self.roi2_high_gray_scan_poll_job is None:
            return

        try:
            self.root.after_cancel(self.roi2_high_gray_scan_poll_job)
        except tk.TclError:
            pass
        finally:
            self.roi2_high_gray_scan_poll_job = None

    def clear_roi2_high_gray_scan_result_queue(self):
        while True:
            try:
                self.roi2_high_gray_scan_result_queue.get_nowait()
            except queue.Empty:
                break

    def invalidate_roi2_high_gray_matches(self, reason=None):
        """清空ROI2高灰度命中结果，并使正在进行的扫描失效"""
        had_matches = bool(self.roi2_high_gray_match_paths)
        was_scanning = self.is_scanning_roi2_high_gray

        self.roi2_high_gray_scan_id += 1
        self.cancel_roi2_high_gray_scan_poll()
        self.clear_roi2_high_gray_scan_result_queue()
        self.is_scanning_roi2_high_gray = False
        self.folder_image_files = []
        self.roi2_high_gray_match_paths = []
        self.roi2_high_gray_match_indices = []
        self.current_roi2_match_position = -1
        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()

        if reason and (had_matches or was_scanning):
            self.status_var.set(reason)

    def on_roi1_or_roi3_config_changed(self):
        """ROI1/ROI3 配置变化时，刷新可视化并清空旧扫描结果"""
        self.invalidate_roi2_high_gray_matches(
            f"ROI1/ROI3配置已变化，请重新扫描 {self.get_roi3_scan_target_text()} 命中结果"
        )
        self.update_roi_visualization()

    def on_roi2_config_changed(self):
        """ROI2 配置变化时，仅刷新当前 ROI2 可视化与对比显示"""
        self.update_roi_visualization()
        self.refresh_roi2_high_gray_comparison()

    def on_roi1_or_roi2_config_changed(self):
        """兼容旧调用，ROI1/ROI3 变化仍按扫描命中失效处理"""
        self.on_roi1_or_roi3_config_changed()

    def get_supported_image_files_in_folder(self, folder_path):
        """获取文件夹中所有支持的图片文件，并按自然顺序排序"""
        image_files = []
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.append(entry.path)

        image_files.sort(key=lambda path: natural_sort_key(os.path.basename(path)))
        return image_files

    def find_folder_image_index(self, file_path):
        """查找当前图片在已扫描文件夹列表中的索引"""
        normalized_target = os.path.normpath(file_path)
        for index, candidate in enumerate(self.folder_image_files):
            if os.path.normpath(candidate) == normalized_target:
                return index
        return -1

    def sync_roi2_high_gray_match_position(self, file_path):
        """根据当前图片路径同步命中位置显示"""
        normalized_target = os.path.normpath(file_path)
        self.current_roi2_match_position = -1

        for index, candidate in enumerate(self.roi2_high_gray_match_paths):
            if os.path.normpath(candidate) == normalized_target:
                self.current_roi2_match_position = index
                break

        self.update_roi2_high_gray_match_ui()

    def get_target_roi2_high_gray_match_position(self, direction):
        """根据当前图片位置查找上一个或下一个命中结果"""
        if not self.roi2_high_gray_match_paths:
            return None

        current_folder_index = self.find_folder_image_index(self.current_roi1_path) if self.current_roi1_path else -1
        total_matches = len(self.roi2_high_gray_match_indices)

        if current_folder_index < 0:
            if 0 <= self.current_roi2_match_position < total_matches:
                return (self.current_roi2_match_position + direction) % total_matches
            return 0 if direction > 0 else total_matches - 1

        if direction > 0:
            for match_position, folder_index in enumerate(self.roi2_high_gray_match_indices):
                if folder_index > current_folder_index:
                    return match_position
            return 0

        for match_position in range(total_matches - 1, -1, -1):
            if self.roi2_high_gray_match_indices[match_position] < current_folder_index:
                return match_position
        return total_matches - 1

    def load_image_from_path(self, file_path, source_label="ROI1图片已加载", reset_frame_diff=False):
        """按路径加载图片，并复用现有的序列检测与界面更新逻辑"""
        try:
            with Image.open(file_path) as image_file:
                self.roi1_image = image_file.copy()

            self.current_roi1_path = file_path
            self.image_path_var.set(f"已加载: {os.path.basename(file_path)}")

            if reset_frame_diff:
                self.prev_frame_avg_gray = None
                self.frame_diff_var.set("--")

            print(f"[DEBUG] 开始检测图片序列: {file_path}")
            self.detect_image_sequence(file_path)
            print(f"[DEBUG] 序列检测完成，找到 {len(self.current_image_sequence)} 张图片")
            print(f"[DEBUG] 当前图片索引: {self.current_image_index}")

            self.update_roi_visualization()
            self.sync_roi2_high_gray_match_position(file_path)
            self.refresh_roi2_high_gray_comparison()

            if self.continuous_check_enabled.get():
                self.root.after(200, self.auto_execute_threshold_processing)

            if self.continuous_heatmap_enabled.get():
                self.root.after(300, self.auto_execute_heatmap_processing)

            zoom_info = f" | Y轴缩放: {int(self.y_zoom_factor * 100)}%" if self.y_zoom_factor != 1.0 else ""
            if len(self.current_image_sequence) > 1 and self.current_image_index >= 0:
                sequence_info = f" | 序列: {self.current_image_index + 1}/{len(self.current_image_sequence)}"
                self.status_var.set(
                    f"{source_label}: {os.path.basename(file_path)}{sequence_info}{zoom_info} (按D/A或→/←切换)"
                )
            else:
                self.status_var.set(f"{source_label}: {os.path.basename(file_path)}{zoom_info}")

            print(f"[INFO] 加载图片: {os.path.basename(file_path)}")
            return True

        except Exception as e:
            print(f"[ERROR] 加载图片失败: {e}")
            self.status_var.set(f"加载图片失败: {str(e)}")
            return False

    def scan_roi2_high_gray_matches(self):
        """扫描当前图片所在文件夹中 ROI2 平均灰度值大于100的图片"""
        self.roi2_compare_mode = "scan"
        self.offline_db_record = None
        self.offline_db_records = []
        self.current_offline_db_position = -1
        self.update_roi2_compare_panel_titles()
        self.update_roi2_high_gray_match_ui()

        if not self.current_roi1_path:
            self.status_var.set("请先导入ROI1图片")
            return

        scan_threshold = self.get_roi3_scan_threshold()
        if scan_threshold is None:
            self.status_var.set("ROI3 扫描阈值必须是 0-255 的整数")
            return

        folder_path = os.path.dirname(self.current_roi1_path)
        image_files = self.get_supported_image_files_in_folder(folder_path)
        if not image_files:
            self.invalidate_roi2_high_gray_matches()
            self.status_var.set("当前文件夹中未找到支持的图片文件")
            return

        self.roi2_high_gray_scan_id += 1
        scan_id = self.roi2_high_gray_scan_id
        self.is_scanning_roi2_high_gray = True
        self.folder_image_files = image_files
        self.roi2_high_gray_match_paths = []
        self.roi2_high_gray_match_indices = []
        self.current_roi2_match_position = -1
        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()
        self.status_var.set(f"正在扫描 ROI3>{scan_threshold}，文件夹内共 {len(image_files)} 张图片...")

        roi_config = self.get_roi_config_values()
        self.cancel_roi2_high_gray_scan_poll()
        self.clear_roi2_high_gray_scan_result_queue()
        scan_thread = threading.Thread(
            target=self._scan_roi2_high_gray_matches_worker,
            args=(scan_id, image_files, roi_config, scan_threshold),
            daemon=True
        )
        scan_thread.start()
        self.roi2_high_gray_scan_poll_job = self.root.after(
            100,
            lambda: self.poll_roi2_high_gray_scan_result(scan_id)
        )

    def _scan_roi2_high_gray_matches_worker(self, scan_id, image_files, roi_config, scan_threshold):
        """后台扫描 ROI2 平均灰度命中图片"""
        matched_paths = []
        matched_indices = []

        for file_index, file_path in enumerate(image_files):
            if scan_id != self.roi2_high_gray_scan_id:
                return

            try:
                with Image.open(file_path) as image_file:
                    roi3_stats = self.compute_roi3_statistics_for_image(image_file, roi_config)

                if roi3_stats is not None and roi3_stats['avg_gray'] > scan_threshold:
                    matched_paths.append(file_path)
                    matched_indices.append(file_index)

            except Exception as e:
                print(f"[WARNING] 扫描图片失败，已跳过: {file_path} ({e})")

        self.roi2_high_gray_scan_result_queue.put({
            'scan_id': scan_id,
            'scan_threshold': scan_threshold,
            'image_files': image_files,
            'matched_paths': matched_paths,
            'matched_indices': matched_indices,
        })

    def poll_roi2_high_gray_scan_result(self, scan_id):
        """鍦ㄤ富绾跨▼杞鎵弿缁撴灉锛岄伩鍏嶅悗鍙扮嚎绋嬭Е纰?Tk"""
        self.roi2_high_gray_scan_poll_job = None

        if scan_id != self.roi2_high_gray_scan_id or not self.is_scanning_roi2_high_gray:
            return

        while True:
            try:
                scan_result = self.roi2_high_gray_scan_result_queue.get_nowait()
            except queue.Empty:
                break

            result_scan_id = scan_result.get('scan_id')
            if result_scan_id != self.roi2_high_gray_scan_id:
                continue

            self.finish_roi2_high_gray_scan(
                result_scan_id,
                scan_result.get('scan_threshold'),
                scan_result.get('image_files', []),
                scan_result.get('matched_paths', []),
                scan_result.get('matched_indices', []),
            )
            return

        self.roi2_high_gray_scan_poll_job = self.root.after(
            100,
            lambda: self.poll_roi2_high_gray_scan_result(scan_id)
        )

    def finish_roi2_high_gray_scan(self, scan_id, scan_threshold, image_files, matched_paths, matched_indices):
        """在主线程中应用扫描结果"""
        if scan_id != self.roi2_high_gray_scan_id:
            return

        self.is_scanning_roi2_high_gray = False
        self.folder_image_files = image_files
        self.roi2_high_gray_match_paths = matched_paths
        self.roi2_high_gray_match_indices = matched_indices
        self.current_roi2_match_position = -1
        self.update_roi2_high_gray_match_ui()

        if not matched_paths:
            self.status_var.set(f"未找到 ROI3 平均灰度 > {scan_threshold} 的图片")
            self.refresh_roi2_high_gray_comparison()
            return

        first_match_path = matched_paths[0]
        if self.load_image_from_path(first_match_path, source_label="ROI3命中图片已加载", reset_frame_diff=True):
            self.current_roi2_match_position = 0
            self.update_roi2_high_gray_match_ui()
            self.refresh_roi2_high_gray_comparison()
            self.status_var.set(
                f"扫描完成：命中 {len(matched_paths)} 张，当前 1/{len(matched_paths)}"
            )

    def goto_previous_roi2_high_gray_match(self):
        """跳转到上一个 ROI2 高灰度命中图片"""
        self.navigate_roi2_high_gray_match(-1)

    def goto_next_roi2_high_gray_match(self):
        """跳转到下一个 ROI2 高灰度命中图片"""
        self.navigate_roi2_high_gray_match(1)

    def get_target_offline_db_position(self, direction):
        total = len(self.offline_db_records)
        if total <= 0:
            return None

        if 0 <= self.current_offline_db_position < total:
            target = self.current_offline_db_position + direction
        else:
            target = 0 if direction > 0 else total - 1

        if target < 0 or target >= total:
            return None

        return target

    def navigate_offline_db_record(self, direction):
        total = len(self.offline_db_records)
        if total <= 0:
            self.status_var.set("请先点击“加载OFFLINE库图”")
            return

        target_position = self.get_target_offline_db_position(direction)
        if target_position is None:
            if direction < 0:
                self.status_var.set("已是最新库图记录")
            else:
                self.status_var.set("已是最旧库图记录")
            return

        try:
            self.apply_offline_db_record_position(target_position)
        except Exception as e:
            msg = f"OFFLINE库图记录无效: {e}"
            self.status_var.set(msg)
            messagebox.showerror("库图记录错误", msg)
            return

        self.update_roi2_compare_panel_titles()
        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()
        self.status_var.set(
            f"OFFLINE库图跳转: {target_position + 1}/{total} | "
            f"ID={self.offline_db_record.get('id')}"
        )

    def navigate_roi2_high_gray_match(self, direction):
        """在 ROI2 高灰度命中图片之间循环切换"""
        if self.roi2_compare_mode == "offline_db":
            self.navigate_offline_db_record(direction)
            return

        if self.is_scanning_roi2_high_gray:
            self.status_var.set("正在扫描，请稍候...")
            return

        if not self.roi2_high_gray_match_paths:
            self.status_var.set(f"请先扫描 {self.get_roi3_scan_target_text()} 命中结果")
            return

        target_position = self.get_target_roi2_high_gray_match_position(direction)
        if target_position is None:
            self.status_var.set("没有可跳转的命中图片")
            return

        target_path = self.roi2_high_gray_match_paths[target_position]
        if self.load_image_from_path(target_path, source_label="ROI3命中图片已加载", reset_frame_diff=True):
            self.current_roi2_match_position = target_position
            self.update_roi2_high_gray_match_ui()
            self.refresh_roi2_high_gray_comparison()
            self.status_var.set(
                f"{self.get_roi3_scan_target_text()} 命中跳转: {target_position + 1}/{len(self.roi2_high_gray_match_paths)}"
            )

    def cancel_roi1_video_extract_poll(self):
        if self.roi1_video_extract_poll_job is None:
            return

        try:
            self.root.after_cancel(self.roi1_video_extract_poll_job)
        except tk.TclError:
            pass
        finally:
            self.roi1_video_extract_poll_job = None

    def clear_roi1_video_extract_result_queue(self):
        while True:
            try:
                self.roi1_video_extract_result_queue.get_nowait()
            except queue.Empty:
                break

    def get_strict_roi1_config_box(self):
        """读取当前 ROI1 坐标；若配置无效则直接报错而不是静默降级。"""
        try:
            roi1_x1 = int(self.roi_vars["roi_capture.default_config.x1"].get().strip())
            roi1_y1 = int(self.roi_vars["roi_capture.default_config.y1"].get().strip())
            roi1_x2 = int(self.roi_vars["roi_capture.default_config.x2"].get().strip())
            roi1_y2 = int(self.roi_vars["roi_capture.default_config.y2"].get().strip())
        except KeyError as e:
            raise ValueError(f"缺少 ROI1 配置项: {e}") from e
        except (ValueError, AttributeError) as e:
            raise ValueError("ROI1 坐标必须填写有效整数后才能进行视频抽帧") from e

        return roi1_x1, roi1_y1, roi1_x2, roi1_y2

    def clamp_roi1_box_to_image(self, image_size, roi1_box):
        """按图像边界裁剪 ROI1 坐标，导出时直接使用当前 ROI1 配置。"""
        image_width, image_height = image_size
        roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi1_box

        roi1_x1 = max(0, min(roi1_x1, image_width - 1))
        roi1_y1 = max(0, min(roi1_y1, image_height - 1))
        roi1_x2 = max(roi1_x1 + 1, min(roi1_x2, image_width))
        roi1_y2 = max(roi1_y1 + 1, min(roi1_y2, image_height))

        if roi1_x2 <= roi1_x1 or roi1_y2 <= roi1_y1:
            raise ValueError("ROI1 坐标超出视频帧范围，无法裁剪出有效区域")

        return roi1_x1, roi1_y1, roi1_x2, roi1_y2

    def extract_roi1_frames_from_video(self):
        """选择视频并每隔 1 秒导出一张 ROI1 裁剪图。"""
        if self.is_extracting_roi1_video:
            self.status_var.set("正在执行视频 ROI1 抽帧，请稍候...")
            return

        try:
            roi1_box = self.get_strict_roi1_config_box()
        except ValueError as e:
            error_message = str(e)
            self.status_var.set(error_message)
            messagebox.showerror("视频抽帧ROI1", error_message, parent=self.root)
            return

        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if not video_path:
            return

        output_dir = filedialog.askdirectory(title="选择 ROI1 抽帧输出文件夹")
        if not output_dir:
            return

        self.roi1_video_extract_task_id += 1
        task_id = self.roi1_video_extract_task_id
        self.is_extracting_roi1_video = True
        self.cancel_roi1_video_extract_poll()
        self.clear_roi1_video_extract_result_queue()

        if self.extract_roi1_video_button is not None:
            self.extract_roi1_video_button.config(state=tk.DISABLED)

        self.status_var.set(
            f"正在从视频每隔 1 秒抽取 ROI1 图像: {os.path.basename(video_path)}"
        )

        worker = threading.Thread(
            target=self._extract_roi1_frames_from_video_worker,
            args=(task_id, video_path, output_dir, roi1_box),
            daemon=True,
        )
        worker.start()
        self.roi1_video_extract_poll_job = self.root.after(
            100,
            lambda: self.poll_roi1_video_extract_result(task_id)
        )

    def extract_roi1_frames_from_video_file(self, video_path, output_dir, roi1_box):
        """从视频中每隔 1 秒抽一帧，并保存该帧的 ROI1 裁剪图。"""
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("缺少 OpenCV(cv2)，无法执行视频抽帧") from e

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            capture.release()
            raise RuntimeError("无法读取视频帧率，无法按每秒抽帧")

        os.makedirs(output_dir, exist_ok=True)

        video_stem = Path(video_path).stem
        next_capture_frame = 0.0
        frame_index = 0
        saved_count = 0
        last_saved_path = None
        clamped_roi1_box = None

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index + 1e-9 >= next_capture_frame:
                    frame_height, frame_width = frame.shape[:2]
                    clamped_roi1_box = self.clamp_roi1_box_to_image(
                        (frame_width, frame_height),
                        roi1_box
                    )
                    roi1_x1, roi1_y1, roi1_x2, roi1_y2 = clamped_roi1_box
                    roi1_frame = frame[roi1_y1:roi1_y2, roi1_x1:roi1_x2]

                    if roi1_frame.size == 0:
                        raise RuntimeError("ROI1 裁剪结果为空，请检查 ROI1 坐标")

                    second_index = saved_count
                    output_name = (
                        f"{video_stem}_roi1_s{second_index:04d}_f{frame_index:06d}.png"
                    )
                    output_path = os.path.join(output_dir, output_name)
                    self.save_video_frame_image(output_path, roi1_frame)

                    last_saved_path = output_path
                    saved_count += 1
                    next_capture_frame += fps

                frame_index += 1
        finally:
            capture.release()

        if frame_index == 0:
            raise RuntimeError("视频中没有可读取的帧")

        if saved_count == 0 or clamped_roi1_box is None:
            raise RuntimeError("未能从视频中抽取任何 ROI1 图像")

        return {
            'video_path': video_path,
            'output_dir': output_dir,
            'saved_count': saved_count,
            'roi1_box': clamped_roi1_box,
            'last_saved_path': last_saved_path,
        }

    def save_video_frame_image(self, output_path, frame_array):
        """保存视频帧图像，兼容 Windows 中文路径。"""
        if frame_array is None or frame_array.size == 0:
            raise RuntimeError("保存 ROI1 图片失败: 空图像")

        try:
            if len(frame_array.shape) == 2:
                image = Image.fromarray(frame_array)
            elif len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                image = Image.fromarray(frame_array[:, :, ::-1])
            elif len(frame_array.shape) == 3 and frame_array.shape[2] == 4:
                image = Image.fromarray(frame_array[:, :, [2, 1, 0, 3]])
            else:
                raise RuntimeError(f"不支持的图像数组形状: {frame_array.shape}")

            image.save(output_path)
        except Exception as e:
            raise RuntimeError(f"保存 ROI1 图片失败: {output_path}") from e

    def _extract_roi1_frames_from_video_worker(self, task_id, video_path, output_dir, roi1_box):
        try:
            result = self.extract_roi1_frames_from_video_file(video_path, output_dir, roi1_box)
            result.update({
                'task_id': task_id,
                'success': True,
            })
        except Exception as e:
            result = {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'video_path': video_path,
                'output_dir': output_dir,
            }

        self.roi1_video_extract_result_queue.put(result)

    def poll_roi1_video_extract_result(self, task_id):
        self.roi1_video_extract_poll_job = None

        if task_id != self.roi1_video_extract_task_id or not self.is_extracting_roi1_video:
            return

        while True:
            try:
                result = self.roi1_video_extract_result_queue.get_nowait()
            except queue.Empty:
                break

            if result.get('task_id') != self.roi1_video_extract_task_id:
                continue

            self.finish_roi1_video_extract(result)
            return

        self.roi1_video_extract_poll_job = self.root.after(
            100,
            lambda: self.poll_roi1_video_extract_result(task_id)
        )

    def finish_roi1_video_extract(self, result):
        if result.get('task_id') != self.roi1_video_extract_task_id:
            return

        self.is_extracting_roi1_video = False
        if self.extract_roi1_video_button is not None:
            self.extract_roi1_video_button.config(state=tk.NORMAL)

        if not result.get('success'):
            self.status_var.set(f"视频 ROI1 抽帧失败: {result.get('error', '未知错误')}")
            return

        roi1_x1, roi1_y1, roi1_x2, roi1_y2 = result['roi1_box']
        self.status_var.set(
            f"视频 ROI1 抽帧完成: 已保存 {result['saved_count']} 张到 {result['output_dir']} "
            f"(ROI1: {roi1_x1},{roi1_y1},{roi1_x2},{roi1_y2})"
        )

    def import_roi1_image(self):
        """导入ROI1图片"""
        filename = filedialog.askopenfilename(
            title="选择ROI1图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.invalidate_roi2_high_gray_matches()
            if not self.load_image_from_path(filename, source_label="ROI1图片已导入", reset_frame_diff=True):
                self.status_var.set("图片导入失败")

    def clear_roi1_image(self):
        """清除ROI1图片"""
        self.invalidate_roi2_high_gray_matches()
        self.roi1_image = None
        self.roi1_photo = None
        self.current_roi1_path = None
        self.current_image_sequence = []
        self.current_image_index = -1
        self.prev_frame_avg_gray = None
        self.frame_diff_var.set("--")
        self.roi2_avg_gray_var.set("--")
        self.roi2_above_threshold_count_var.set("--")
        self.image_path_var.set("未选择图片")
        self.roi_canvas.delete("all")
        self.pixel_info_var.set("请导入图片并在ROI1区域移动鼠标")  # 重置固定文本框
        self.update_roi2_high_gray_match_ui()
        self.refresh_roi2_high_gray_comparison()
        self.status_var.set("ROI1图片已清除")

    def get_roi_config_values(self):
        """获取ROI配置值"""
        try:
            # ROI1配置
            roi1_x1 = int(self.roi_vars.get("roi_capture.default_config.x1", tk.StringVar()).get() or 0)
            roi1_y1 = int(self.roi_vars.get("roi_capture.default_config.y1", tk.StringVar()).get() or 0)
            roi1_x2 = int(self.roi_vars.get("roi_capture.default_config.x2", tk.StringVar()).get() or 100)
            roi1_y2 = int(self.roi_vars.get("roi_capture.default_config.y2", tk.StringVar()).get() or 100)

            # ROI2扩展参数
            roi2_left = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.left", tk.StringVar()).get() or 20)
            roi2_right = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.right", tk.StringVar()).get() or 30)
            roi2_top = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.top", tk.StringVar()).get() or 60)
            roi2_bottom = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.bottom", tk.StringVar()).get() or 20)

            # ROI3扩展参数
            roi3_left = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.left", tk.StringVar()).get() or 30)
            roi3_right = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.right", tk.StringVar()).get() or 40)
            roi3_top = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.top", tk.StringVar()).get() or 70)
            roi3_bottom = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.bottom", tk.StringVar()).get() or 30)

            roi_config = {
                'roi1': (roi1_x1, roi1_y1, roi1_x2, roi1_y2),
                'roi2': (roi2_left, roi2_right, roi2_top, roi2_bottom),
                'roi3': (roi3_left, roi3_right, roi3_top, roi3_bottom)
            }

            # 验证ROI3配置（如果有ROI1图像）
            if self.roi1_image:
                roi1_size = self.roi1_image.size
                roi_config = self.validate_roi3_configuration(roi_config, roi1_size)

            return roi_config
        except (ValueError, AttributeError):
            # 如果配置无效，返回默认值
            return {
                'roi1': (0, 0, 100, 100),
                'roi2': (20, 30, 60, 20),
                'roi3': (30, 40, 70, 30)
            }

    def validate_roi3_configuration(self, roi_config, roi1_size):
        """验证和调整ROI3配置以匹配ROI1尺寸"""
        try:
            roi1_width, roi1_height = roi1_size
            center_x, center_y = roi1_width // 2, roi1_height // 2

            # 确保roi3参数存在
            if 'roi3' not in roi_config:
                print("[WARNING] ROI3配置缺失，使用默认值")
                roi_config['roi3'] = (30, 40, 70, 30)
                return roi_config

            roi3_params = roi_config['roi3']
            left, right, top, bottom = roi3_params

            # 检查ROI3是否会超出ROI1边界
            total_width = left + right
            total_height = top + bottom

            max_safe_width = min(center_x, roi1_width - center_x) * 2
            max_safe_height = min(center_y, roi1_height - center_y) * 2

            adjustments_made = []

            # 自动调整超出边界的参数
            if total_width > max_safe_width and max_safe_width > 0:
                scale_factor = max_safe_width / total_width
                left = int(left * scale_factor)
                right = int(right * scale_factor)
                adjustments_made.append(f"宽度缩放至{scale_factor:.2f}")

            if total_height > max_safe_height and max_safe_height > 0:
                scale_factor = max_safe_height / total_height
                top = int(top * scale_factor)
                bottom = int(bottom * scale_factor)
                adjustments_made.append(f"高度缩放至{scale_factor:.2f}")

            # 确保最小尺寸
            min_size = 10
            if left + right < min_size:
                left = max(5, left)
                right = max(5, right)
                adjustments_made.append("应用最小宽度")

            if top + bottom < min_size:
                top = max(5, top)
                bottom = max(5, bottom)
                adjustments_made.append("应用最小高度")

            if adjustments_made:
                print(f"[INFO] ROI3配置已自动调整: {', '.join(adjustments_made)}")
                print(f"[INFO] 调整后ROI3参数: left={left}, right={right}, top={top}, bottom={bottom}")

                # 更新配置（可选，如果需要保存）
                roi_config['roi3'] = (left, right, top, bottom)
            else:
                print(f"[DEBUG] ROI3配置验证通过: left={left}, right={right}, top={top}, bottom={bottom}")

            return roi_config

        except Exception as e:
            print(f"[ERROR] ROI3配置验证失败: {e}")
            import traceback
            traceback.print_exc()
            return roi_config

    def compute_roi_layout(self, image_size, roi_config):
        """计算 ROI1/ROI2/ROI3 的统一几何布局，供预览和扫描复用"""
        original_img_width, original_img_height = image_size

        roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']
        roi1_x1 = max(0, min(roi1_x1, original_img_width - 1))
        roi1_y1 = max(0, min(roi1_y1, original_img_height - 1))
        roi1_x2 = max(roi1_x1 + 1, min(roi1_x2, original_img_width))
        roi1_y2 = max(roi1_y1 + 1, min(roi1_y2, original_img_height))

        roi1_width = roi1_x2 - roi1_x1
        roi1_height = roi1_y2 - roi1_y1
        min_size = 50

        img_area = original_img_width * original_img_height
        roi1_area = roi1_width * roi1_height

        if roi1_area < img_area * 0.25:
            margin_x = int(original_img_width * 0.1)
            margin_y = int(original_img_height * 0.1)
            roi1_x1 = margin_x
            roi1_y1 = margin_y
            roi1_x2 = original_img_width - margin_x
            roi1_y2 = original_img_height - margin_y
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1
        else:
            if roi1_width < min_size:
                center_x_abs = (roi1_x1 + roi1_x2) // 2
                roi1_x1 = max(0, center_x_abs - min_size // 2)
                roi1_x2 = min(original_img_width, roi1_x1 + min_size)
                roi1_width = roi1_x2 - roi1_x1

            if roi1_height < min_size:
                center_y_abs = (roi1_y1 + roi1_y2) // 2
                roi1_y1 = max(0, center_y_abs - min_size // 2)
                roi1_y2 = min(original_img_height, roi1_y1 + min_size)
                roi1_height = roi1_y2 - roi1_y1

        center_x = roi1_width // 2
        center_y = roi1_height // 2

        roi2_left = center_x - roi_config['roi2'][0]
        roi2_top = center_y - roi_config['roi2'][2]
        roi2_right = center_x + roi_config['roi2'][1]
        roi2_bottom = center_y + roi_config['roi2'][3]

        roi3_left = center_x - roi_config['roi3'][0]
        roi3_top = center_y - roi_config['roi3'][2]
        roi3_right = center_x + roi_config['roi3'][1]
        roi3_bottom = center_y + roi_config['roi3'][3]

        return {
            'original_img_width': original_img_width,
            'original_img_height': original_img_height,
            'roi1_x1': roi1_x1,
            'roi1_y1': roi1_y1,
            'roi1_x2': roi1_x2,
            'roi1_y2': roi1_y2,
            'roi1_width': roi1_width,
            'roi1_height': roi1_height,
            'center_x': center_x,
            'center_y': center_y,
            'roi2_left': roi2_left,
            'roi2_top': roi2_top,
            'roi2_right': roi2_right,
            'roi2_bottom': roi2_bottom,
            'roi2_width': roi2_right - roi2_left,
            'roi2_height': roi2_bottom - roi2_top,
            'roi3_left': roi3_left,
            'roi3_top': roi3_top,
            'roi3_right': roi3_right,
            'roi3_bottom': roi3_bottom,
            'roi3_width': roi3_right - roi3_left,
            'roi3_height': roi3_bottom - roi3_top,
        }

    def extract_roi2_image_and_stats(self, image, roi_config, threshold=None, include_full_image=False):
        layout = self.compute_roi_layout(image.size, roi_config)
        roi1_region = image.crop((
            layout['roi1_x1'],
            layout['roi1_y1'],
            layout['roi1_x2'],
            layout['roi1_y2']
        ))

        roi_x = max(0, layout['roi2_left'])
        roi_y = max(0, layout['roi2_top'])
        roi_x2 = min(roi1_region.size[0], layout['roi2_left'] + layout['roi2_width'])
        roi_y2 = min(roi1_region.size[1], layout['roi2_top'] + layout['roi2_height'])

        if roi_x2 <= roi_x or roi_y2 <= roi_y:
            return None

        roi2_region = roi1_region.crop((roi_x, roi_y, roi_x2, roi_y2)).copy()
        roi2_stats = self.compute_roi_gray_statistics(
            roi2_region,
            0,
            0,
            roi2_region.size[0],
            roi2_region.size[1],
            threshold
        )

        return {
            'roi2_image': roi2_region,
            'stats': roi2_stats,
            'layout': layout,
            'full_image': image.copy() if include_full_image else None,
            'roi2_box_in_full_image': (
                layout['roi1_x1'] + roi_x,
                layout['roi1_y1'] + roi_y,
                layout['roi1_x1'] + roi_x2,
                layout['roi1_y1'] + roi_y2,
            ),
        }

    def load_roi2_image_payload_from_path(self, file_path, roi_config, threshold=None, include_full_image=False):
        try:
            with Image.open(file_path) as image_file:
                return self.extract_roi2_image_and_stats(
                    image_file,
                    roi_config,
                    threshold,
                    include_full_image=include_full_image
                )
        except Exception as e:
            print(f"[WARNING] 加载 ROI2 对比图失败，已跳过: {file_path} ({e})")
            return None

    def compute_roi2_statistics_for_image(self, image, roi_config, threshold=None):
        """按当前 ROI1/ROI2 配置计算单张图片的 ROI2 灰度统计"""
        roi2_payload = self.extract_roi2_image_and_stats(image, roi_config, threshold)
        if roi2_payload is None:
            return None
        return roi2_payload['stats']

    def extract_roi3_image_and_stats(self, image, roi_config, threshold=None):
        layout = self.compute_roi_layout(image.size, roi_config)
        roi1_region = image.crop((
            layout['roi1_x1'],
            layout['roi1_y1'],
            layout['roi1_x2'],
            layout['roi1_y2']
        ))

        roi_x = max(0, layout['roi3_left'])
        roi_y = max(0, layout['roi3_top'])
        roi_x2 = min(roi1_region.size[0], layout['roi3_left'] + layout['roi3_width'])
        roi_y2 = min(roi1_region.size[1], layout['roi3_top'] + layout['roi3_height'])

        if roi_x2 <= roi_x or roi_y2 <= roi_y:
            return None

        roi3_region = roi1_region.crop((roi_x, roi_y, roi_x2, roi_y2)).copy()
        roi3_stats = self.compute_roi_gray_statistics(
            roi3_region,
            0,
            0,
            roi3_region.size[0],
            roi3_region.size[1],
            threshold
        )

        return {
            'roi3_image': roi3_region,
            'stats': roi3_stats,
            'layout': layout,
        }

    def compute_roi3_statistics_for_image(self, image, roi_config, threshold=None):
        """按当前 ROI1/ROI3 配置计算单张图片的 ROI3 灰度统计"""
        roi3_payload = self.extract_roi3_image_and_stats(image, roi_config, threshold)
        if roi3_payload is None:
            return None
        return roi3_payload['stats']

    def update_roi_visualization(self):
        """更新ROI可视化"""
        # 清除画布
        self.roi_canvas.delete("all")

        if self.roi1_image is None:
            self.roi2_avg_gray_var.set("--")
            self.roi2_above_threshold_count_var.set("--")
            # 如果没有图片，显示提示文本
            self.roi_canvas.create_text(
                320, 450,  # 居中在640x900画布中
                text="请导入ROI1图片以预览ROI区域\n\n支持格式：PNG, JPG, JPEG, BMP, TIFF",
                fill='gray',
                font=('Arial', 14),
                justify=tk.CENTER
            )
            return

        # 获取ROI配置
        roi_config = self.get_roi_config_values()

        # 获取画布尺寸
        canvas_width = self.roi_canvas.winfo_width()
        canvas_height = self.roi_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # 画布还未初始化，使用默认尺寸 (640x900)
            canvas_width = 640
            canvas_height = 900

        layout = self.compute_roi_layout(self.roi1_image.size, roi_config)
        original_img_width = layout['original_img_width']
        original_img_height = layout['original_img_height']
        roi1_x1 = layout['roi1_x1']
        roi1_y1 = layout['roi1_y1']
        roi1_x2 = layout['roi1_x2']
        roi1_y2 = layout['roi1_y2']
        roi1_width = layout['roi1_width']
        roi1_height = layout['roi1_height']

        # 从原始图片中提取ROI1区域作为背景
        roi1_avg_gray = self.compute_roi_grayscale(self.roi1_image, roi1_x1, roi1_y1, roi1_width, roi1_height)
        roi1_region = self.roi1_image.crop((roi1_x1, roi1_y1, roi1_x2, roi1_y2))

        # 计算缩放比例以适应画布
        scale_x = (canvas_width - 20) / roi1_width
        scale_y = (canvas_height - 20) / roi1_height
        base_scale = min(scale_x, scale_y, 1.0)  # 不放大图片的基础缩放

        # 应用ROI1缩放因子
        display_scale = base_scale * self.roi1_zoom_factor.get()

        # 缩放ROI1图片
        display_width = int(roi1_width * display_scale)
        display_height = int(roi1_height * display_scale)
        display_image = roi1_region.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # 转换为PhotoImage并显示
        self.roi1_photo = ImageTk.PhotoImage(display_image)

        # 计算居中位置和基本偏移量
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        x_offset = (canvas_width - display_width) // 2
        y_offset = (canvas_height - display_height) // 2

        # 应用平移偏移量
        pan_offset_x = self.roi1_pan_offset_x.get()
        pan_offset_y = self.roi1_pan_offset_y.get()

        final_x = center_x + pan_offset_x
        final_y = center_y + pan_offset_y

        self.roi_canvas.create_image(
            final_x, final_y,
            image=self.roi1_photo,
            anchor=tk.CENTER
        )

        # 更新x_offset和y_offset以包含平移偏移量（用于ROI绘制）
        x_offset += pan_offset_x
        y_offset += pan_offset_y

        # ROI1中心点作为交点（在ROI1坐标系中的坐标）
        center_x = layout['center_x']
        center_y = layout['center_y']
        roi2_left = layout['roi2_left']
        roi2_top = layout['roi2_top']
        roi2_right = layout['roi2_right']
        roi2_bottom = layout['roi2_bottom']
        roi2_width = layout['roi2_width']
        roi2_height = layout['roi2_height']
        roi3_left = layout['roi3_left']
        roi3_top = layout['roi3_top']
        roi3_right = layout['roi3_right']
        roi3_bottom = layout['roi3_bottom']
        roi3_width = layout['roi3_width']
        roi3_height = layout['roi3_height']

        roi2_threshold = self.get_roi2_stats_threshold()
        roi2_payload = self.extract_roi2_image_and_stats(self.roi1_image, roi_config, roi2_threshold)
        roi2_stats = roi2_payload['stats'] if roi2_payload is not None else None

        roi2_avg_gray_text = "ROI2平均灰度: --"
        roi2_threshold_count_text = "ROI2 > 阈值像素: --"
        if roi2_stats is not None:
            roi2_avg_gray_text = f"ROI2平均灰度: {roi2_stats['avg_gray']:.2f}"
            self.roi2_avg_gray_var.set(f"{roi2_stats['avg_gray']:.2f}")

            if roi2_threshold is None:
                roi2_threshold_count_text = "ROI2 > 阈值像素: 阈值无效 (需填0-255)"
                self.roi2_above_threshold_count_var.set("阈值无效")
            else:
                roi2_threshold_count_text = f"ROI2 > {roi2_threshold}: {roi2_stats['pixels_above_threshold']} 像素"
                self.roi2_above_threshold_count_var.set(str(roi2_stats['pixels_above_threshold']))
        else:
            self.roi2_avg_gray_var.set("--")
            self.roi2_above_threshold_count_var.set("--")

        # 转换ROI2、ROI3坐标到显示坐标系
        roi2_img_left = (roi2_left * display_scale) + x_offset
        roi2_img_top = (roi2_top * display_scale) + y_offset
        roi2_img_right = (roi2_right * display_scale) + x_offset
        roi2_img_bottom = (roi2_bottom * display_scale) + y_offset

        roi3_img_left = (roi3_left * display_scale) + x_offset
        roi3_img_top = (roi3_top * display_scale) + y_offset
        roi3_img_right = (roi3_right * display_scale) + x_offset
        roi3_img_bottom = (roi3_bottom * display_scale) + y_offset

        # 交点坐标（ROI1中心点在显示坐标系中的位置）
        intersection_x = (center_x * display_scale) + x_offset
        intersection_y = (center_y * display_scale) + y_offset

        # ROI1边界（显示坐标，即背景图片边界）
        roi1_img_left = x_offset
        roi1_img_top = y_offset
        roi1_img_right = x_offset + display_width
        roi1_img_bottom = y_offset + display_height

        # 绘制ROI3区域（蓝色虚线，叠加在背景上）
        self.roi_canvas.create_rectangle(
            roi3_img_left, roi3_img_top, roi3_img_right, roi3_img_bottom,
            outline='blue', width=2, dash=(8, 4)
        )

        # 绘制ROI2区域（红色实线，叠加在背景上）
        self.roi_canvas.create_rectangle(
            roi2_img_left, roi2_img_top, roi2_img_right, roi2_img_bottom,
            outline='red', width=3
        )

        # 绘制交点（ROI1中心点，绿色圆点，叠加在背景上）
        self.roi_canvas.create_oval(
            intersection_x - 6, intersection_y - 6,
            intersection_x + 6, intersection_y + 6,
            fill='lime', outline='darkgreen', width=2
        )

        # 添加标签
        self.roi_canvas.create_text(
            roi2_img_left - 5, roi2_img_top - 5,
            text="ROI2", fill='red', font=('Arial', 10, 'bold'), anchor='se'
        )

        self.roi_canvas.create_text(
            roi3_img_left - 5, roi3_img_top - 5,
            text="ROI3", fill='blue', font=('Arial', 10, 'bold'), anchor='se'
        )

        # 添加ROI1标签
        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_top + 5,
            text="ROI1(背景)", fill='darkgreen', font=('Arial', 10, 'bold'), anchor='nw'
        )

        # 添加尺寸信息
        # ROI1 预览区域：显示 ROI1 平均灰度值（紫色文字）
        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_top + 25,
            text=f"ROI1平均灰度: {roi1_avg_gray:.2f}",
            fill='#800080', font=('Arial', 10, 'bold'), anchor='nw'
        )

        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_top + 45,
            text=roi2_avg_gray_text,
            fill='red', font=('Arial', 10, 'bold'), anchor='nw'
        )

        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_top + 65,
            text=roi2_threshold_count_text,
            fill='red', font=('Arial', 10, 'bold'), anchor='nw'
        )

        # 在ROI2区域右下角显示尺寸
        self.roi_canvas.create_text(
            roi2_img_right - 5, roi2_img_bottom - 5,
            text=f"{roi2_width}x{roi2_height}",
            fill='red', font=('Arial', 8), anchor='se'
        )

        # 在ROI3区域右下角显示尺寸
        self.roi_canvas.create_text(
            roi3_img_right - 5, roi3_img_bottom - 5,
            text=f"{roi3_width}x{roi3_height}",
            fill='blue', font=('Arial', 8), anchor='se'
        )

        # 在ROI1区域左下角显示图片信息
        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_bottom - 5,
            text=f"原图: {original_img_width}x{original_img_height} | ROI1: {roi1_width}x{roi1_height} | 缩放: {display_scale:.2f}x",
            fill='darkgreen', font=('Arial', 8), anchor='sw'
        )

        # 绘制ROI2和ROI3的灰度曲线图
        self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

    def compute_roi_grayscale(self, roi1_image, roi_x, roi_y, roi_width, roi_height):
        """计算ROI区域的平均灰度值"""
        try:
            # 裁剪ROI区域
            roi_region = roi1_image.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))

            # 转换为灰度图像
            if roi_region.mode != 'L':
                roi_region = roi_region.convert('L')

            # 计算平均灰度值
            import numpy as np
            roi_array = np.array(roi_region)
            avg_gray = float(np.mean(roi_array))

            return avg_gray
        except Exception as e:
            print(f"计算ROI灰度值失败: {e}")
            return 0.0

    def get_roi2_stats_threshold(self):
        """获取ROI2统计使用的像素阈值"""
        try:
            raw_value = self.roi2_stats_threshold_var.get().strip()
            if not raw_value:
                return None

            threshold = int(raw_value)
            if 0 <= threshold <= 255:
                return threshold

            return None
        except (ValueError, tk.TclError, AttributeError):
            return None

    def compute_roi_gray_statistics(self, roi_image, roi_x, roi_y, roi_width, roi_height, threshold=None):
        """计算ROI区域的平均灰度值和大于阈值的像素数量"""
        try:
            img_width, img_height = roi_image.size
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_x2 = min(img_width, roi_x + roi_width)
            roi_y2 = min(img_height, roi_y + roi_height)

            actual_width = roi_x2 - roi_x
            actual_height = roi_y2 - roi_y
            if actual_width <= 0 or actual_height <= 0:
                return None

            roi_region = roi_image.crop((roi_x, roi_y, roi_x2, roi_y2))
            if roi_region.mode != 'L':
                roi_region = roi_region.convert('L')

            roi_array = np.array(roi_region)
            avg_gray = float(np.mean(roi_array))

            pixels_above_threshold = None
            if threshold is not None:
                pixels_above_threshold = int(np.sum(roi_array > threshold))

            return {
                'avg_gray': avg_gray,
                'pixels_above_threshold': pixels_above_threshold,
                'total_pixels': int(roi_array.size)
            }
        except Exception as e:
            print(f"计算ROI灰度统计失败: {e}")
            return None

    def draw_grayscale_curves(self, roi_config, img_width, img_height):
        """绘制ROI2和ROI3的灰度直方图（像素分布曲线）"""
        # 清除曲线画布
        self.curve_canvas.delete("all")

        if self.roi1_image is None:
            self.curve_canvas.create_text(
                300, 450,  # 居中在600x900画布中
                text="请导入ROI1图片以显示灰度直方图\n\nX轴：灰度值(0-255)\nY轴：像素数量",
                fill='gray',
                font=('Arial', 14),
                justify=tk.CENTER
            )
            return

        try:
            # 获取画布尺寸
            canvas_width = self.curve_canvas.winfo_width()
            canvas_height = self.curve_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 900

            # 设置绘图区域（为600x900画布优化边距）
            margin_left = 80
            margin_right = 40
            margin_top = 40
            margin_bottom = 100

            plot_width = canvas_width - margin_left - margin_right
            plot_height = canvas_height - margin_top - margin_bottom

            # 绘制坐标轴
            # X轴（灰度值）
            self.curve_canvas.create_line(
                margin_left, margin_top + plot_height,
                margin_left + plot_width, margin_top + plot_height,
                fill='black', width=2
            )
            # Y轴（像素个数）
            self.curve_canvas.create_line(
                margin_left, margin_top,
                margin_left, margin_top + plot_height,
                fill='black', width=2
            )

            # 添加坐标轴标签
            self.curve_canvas.create_text(
                margin_left + plot_width // 2, canvas_height - 15,
                text="X: 灰度值(0-255) / 列号 | Y: 像素数 / 灰度值", fill='black', font=('Arial', 12, 'bold')
            )
            self.curve_canvas.create_text(
                25, margin_top + plot_height // 2,
                text="像素个数", fill='black', font=('Arial', 12, 'bold'), angle=90
            )

            # 获取ROI1实际裁剪区域（与update_roi_visualization保持一致）
            roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1

            # 获取原始图片尺寸
            original_img_width, original_img_height = img_width, img_height

            # 确保ROI1坐标在有效范围内
            roi1_x1 = max(0, min(roi1_x1, original_img_width - 1))
            roi1_y1 = max(0, min(roi1_y1, original_img_height - 1))
            roi1_x2 = max(roi1_x1 + 1, min(roi1_x2, original_img_width))
            roi1_y2 = max(roi1_y1 + 1, min(roi1_y2, original_img_height))

            # 重新计算ROI1实际尺寸
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1

            # 如果ROI1区域相对于原图太小，则使用图片的大部分区域（与update_roi_visualization保持一致）
            min_size = 50
            img_area = original_img_width * original_img_height
            roi1_area = roi1_width * roi1_height

            if roi1_area < img_area * 0.25:  # 如果ROI1面积小于原图面积的25%
                # 使用图片的80%作为ROI1区域，居中显示
                margin_x = int(original_img_width * 0.1)
                margin_y = int(original_img_height * 0.1)
                roi1_x1 = margin_x
                roi1_y1 = margin_y
                roi1_x2 = original_img_width - margin_x
                roi1_y2 = original_img_height - margin_y
                roi1_width = roi1_x2 - roi1_x1
                roi1_height = roi1_y2 - roi1_y1
            else:
                # 否则应用最小尺寸逻辑
                if roi1_width < min_size:
                    # 居中扩展ROI1宽度
                    center_x_abs = (roi1_x1 + roi1_x2) // 2
                    roi1_x1 = max(0, center_x_abs - min_size // 2)
                    roi1_x2 = min(original_img_width, roi1_x1 + min_size)
                    roi1_width = roi1_x2 - roi1_x1

                if roi1_height < min_size:
                    # 居中扩展ROI1高度
                    center_y_abs = (roi1_y1 + roi1_y2) // 2
                    roi1_y1 = max(0, center_y_abs - min_size // 2)
                    roi1_y2 = min(original_img_height, roi1_y1 + min_size)
                    roi1_height = roi1_y2 - roi1_y1

            # ROI1中心点（相对于ROI1裁剪区域）
            center_x = roi1_width // 2
            center_y = roi1_height // 2

            # ROI2和ROI3的扩展参数
            roi2_left, roi2_right, roi2_top, roi2_bottom = roi_config['roi2']
            roi3_left, roi3_right, roi3_top, roi3_bottom = roi_config['roi3']

            # 计算ROI2和ROI3相对于ROI1裁剪区域的坐标
            roi2_x = center_x - roi2_left
            roi2_y = center_y - roi2_top
            roi2_width = roi2_left + roi2_right
            roi2_height = roi2_top + roi2_bottom

            roi3_x = center_x - roi3_left
            roi3_y = center_y - roi3_top
            roi3_width = roi3_left + roi3_right
            roi3_height = roi3_top + roi3_bottom

            # 创建ROI1裁剪区域用于直方图计算
            roi1_region = self.roi1_image.crop((roi1_x1, roi1_y1, roi1_x2, roi1_y2))

            # 获取ROI2和ROI3的灰度直方图数据（使用ROI1裁剪区域）
            roi2_histogram = self.compute_grayscale_histogram(
                roi1_region, roi2_x, roi2_y, roi2_width, roi2_height
            )
            roi3_histogram = self.compute_grayscale_histogram(
                roi1_region, roi3_x, roi3_y, roi3_width, roi3_height
            )

            # 调试输出
            print(f"[DEBUG] ROI2区域: x={roi2_x}, y={roi2_y}, w={roi2_width}, h={roi2_height}")
            print(f"[DEBUG] ROI3区域: x={roi3_x}, y={roi3_y}, w={roi3_width}, h={roi3_height}")
            print(f"[DEBUG] ROI2直方图长度: {len(roi2_histogram) if roi2_histogram else 0}")
            print(f"[DEBUG] ROI3直方图长度: {len(roi3_histogram) if roi3_histogram else 0}")
            if roi2_histogram:
                roi2_total = sum(roi2_histogram)
                roi2_max = max(roi2_histogram)
                print(f"[DEBUG] ROI2总像素: {roi2_total}, 最大像素数: {roi2_max}")
            if roi3_histogram:
                roi3_total = sum(roi3_histogram)
                roi3_max = max(roi3_histogram)
                print(f"[DEBUG] ROI3总像素: {roi3_total}, 最大像素数: {roi3_max}")

            # 确定Y轴范围（像素个数）- 强制使用500作为默认最大值
            roi2_max_count = max(roi2_histogram) if roi2_histogram else 1
            roi3_max_count = max(roi3_histogram) if roi3_histogram else 1
            base_max_count = 500  # 固定使用500作为Y轴默认最大值，忽略实际ROI像素数

            print(f"[INFO] 强制设置Y轴默认范围为0-{base_max_count}像素")
            print(f"[DEBUG] ROI2实际最大像素数: {roi2_max_count} (可能被截断)")
            print(f"[DEBUG] ROI3实际最大像素数: {roi3_max_count} (可能被截断)")

            # 应用Y轴缩放因子
            max_count = int(base_max_count / self.y_zoom_factor)  # 缩放因子越大，显示的Y轴范围越小（放大）

            print(f"[DEBUG] ROI2最大像素数: {roi2_max_count}")
            print(f"[DEBUG] ROI3最大像素数: {roi3_max_count}")
            print(f"[DEBUG] 基础最大值: {base_max_count}")
            print(f"[DEBUG] Y轴缩放因子: {self.y_zoom_factor}")
            print(f"[DEBUG] 最终Y轴最大值: {max_count}")

            # 绘制ROI2灰度直方图（红色）
            roi2_points = []
            for gray_val in range(256):
                if gray_val < len(roi2_histogram):
                    count = roi2_histogram[gray_val]
                    canvas_x = margin_left + (gray_val * plot_width // 255)
                    # 限制count不超过max_count，防止曲线超出Y轴范围
                    display_count = min(count, max_count)
                    canvas_y = margin_top + plot_height - (display_count * plot_height // max_count)
                    roi2_points.extend([canvas_x, canvas_y])

            # 使用平滑线绘制直方图
            if len(roi2_points) >= 4:
                self.curve_canvas.create_line(
                    roi2_points, fill='red', width=2, smooth=False
                )

            # 绘制ROI3灰度直方图（蓝色）
            roi3_points = []
            for gray_val in range(256):
                if gray_val < len(roi3_histogram):
                    count = roi3_histogram[gray_val]
                    canvas_x = margin_left + (gray_val * plot_width // 255)
                    # 限制count不超过max_count，防止曲线超出Y轴范围
                    display_count = min(count, max_count)
                    canvas_y = margin_top + plot_height - (display_count * plot_height // max_count)
                    roi3_points.extend([canvas_x, canvas_y])

            if len(roi3_points) >= 4:
                self.curve_canvas.create_line(
                    roi3_points, fill='blue', width=2, smooth=False
                )

            # ========== 新增：绘制ROI3列平均灰度值曲线 ==========
            # 从ROI1裁剪区域中提取ROI3并计算列平均灰度值
            try:
                # 确保ROI3坐标在有效范围内
                roi3_x_clamped = max(0, min(roi3_x, roi1_width - 1))
                roi3_y_clamped = max(0, min(roi3_y, roi1_height - 1))
                roi3_x2 = min(roi1_width, roi3_x + roi3_width)
                roi3_y2 = min(roi1_height, roi3_y + roi3_height)
                roi3_width_actual = roi3_x2 - roi3_x_clamped
                roi3_height_actual = roi3_y2 - roi3_y_clamped

                if roi3_width_actual > 0 and roi3_height_actual > 0:
                    # 从ROI1裁剪区域中提取ROI3
                    roi3_region = roi1_region.crop((
                        roi3_x_clamped,
                        roi3_y_clamped,
                        roi3_x2,
                        roi3_y2
                    ))

                    # 计算列平均灰度值
                    column_means = self.compute_roi3_column_mean_gray(roi3_region)

                    # 绘制列平均灰度值曲线（绿色虚线）
                    if column_means and len(column_means) > 0:
                        column_curve_points = []
                        num_columns = len(column_means)

                        # 计算列平均灰度值的最大值、最小值和差值
                        max_mean_gray = max(column_means)
                        min_mean_gray = min(column_means)
                        diff_mean_gray = max_mean_gray - min_mean_gray

                        # 更新差值显示标签（保留2位小数）
                        self.column_mean_diff_var.set(f"{diff_mean_gray:.2f}")

                        for col_idx, mean_gray in enumerate(column_means):
                            # X轴：列号直接映射（0到num_columns-1映射到0到plot_width）
                            if num_columns > 1:
                                canvas_x = margin_left + (col_idx * plot_width // (num_columns - 1))
                            else:
                                canvas_x = margin_left + plot_width // 2

                            # Y轴：灰度值直接映射（0-255映射到0-max_count）
                            # 这样灰度值128会显示在Y轴中间位置
                            # 注意：这里将灰度值0-255映射到Y轴的0-max_count范围
                            canvas_y = margin_top + plot_height - (mean_gray * plot_height // max_count)

                            column_curve_points.extend([canvas_x, canvas_y])

                        # 使用平滑线绘制
                        if len(column_curve_points) >= 4:  # 至少2个点
                            self.curve_canvas.create_line(
                                column_curve_points,
                                fill='green',
                                width=2,
                                smooth=True,
                                dash=(5, 3)  # 虚线样式，便于区分
                            )

                            # 在曲线终点标注列数
                            last_x = column_curve_points[-2]
                            last_y = column_curve_points[-1]
                            self.curve_canvas.create_text(
                                last_x + 10, last_y,
                                text=f"列平均({num_columns}列)",
                                fill='green',
                                font=('Arial', 9),
                                anchor='w'
                            )

                            print(f"[INFO] ROI3列平均曲线已绘制: {num_columns}列")

            except Exception as e:
                print(f"[ERROR] 绘制ROI3列平均灰度值曲线失败: {e}")
                import traceback
                traceback.print_exc()
            # ========== 新增代码结束 ==========

            # 添加X轴刻度（灰度值）
            x_ticks = [0, 64, 128, 192, 255]
            for gray_val in x_ticks:
                canvas_x = margin_left + (gray_val * plot_width // 255)
                self.curve_canvas.create_line(
                    canvas_x, margin_top + plot_height,
                    canvas_x, margin_top + plot_height + 8,
                    fill='black', width=1
                )
                self.curve_canvas.create_text(
                    canvas_x, margin_top + plot_height + 20,
                    text=str(gray_val), fill='black', font=('Arial', 10, 'bold')
                )

            # 添加Y轴刻度（像素个数）
            y_ticks = [0, max_count//4, max_count//2, max_count*3//4, max_count]
            for i, count in enumerate(y_ticks):
                canvas_y = margin_top + plot_height - (count * plot_height // max_count)
                self.curve_canvas.create_line(
                    margin_left - 5, canvas_y, margin_left, canvas_y,
                    fill='black', width=1
                )
                self.curve_canvas.create_text(
                    margin_left - 15, canvas_y,
                    text=f"{count}", fill='black', font=('Arial', 10, 'bold'),
                    anchor='e'
                )

            # 注释：灰度直方图不需要显示峰值检测阈值线
            # 阈值线适用于信号处理图表，不适用于像素分布直方图

            # 添加统计信息和缩放级别
            roi2_total = sum(roi2_histogram) if roi2_histogram else 0
            roi3_total = sum(roi3_histogram) if roi3_histogram else 0
            roi2_avg = (
                sum(gray_val * count for gray_val, count in enumerate(roi2_histogram)) / roi2_total
                if roi2_total > 0 else 0
            )
            roi3_avg = (
                sum(gray_val * count for gray_val, count in enumerate(roi3_histogram)) / roi3_total
                if roi3_total > 0 else 0
            )

            # 获取缩放信息
            zoom_info = self.get_y_zoom_info()
            zoom_text = f"Y轴缩放: {zoom_info['zoom_percentage']}%" if zoom_info['is_zoomed'] else "Y轴缩放: 100%"

            stats_text = f"ROI2: 总像素{roi2_total} 平均{roi2_avg:.1f}  |  ROI3: 总像素{roi3_total} 平均{roi3_avg:.1f}  |  {zoom_text}"
            self.curve_canvas.create_text(
                canvas_width // 2, 20,
                text=stats_text, fill='black', font=('Arial', 11, 'bold')
            )

            # 添加缩放操作提示
            if self.roi1_image:  # 只在有图片时显示提示
                hint_text = "提示: 使用鼠标滚轮可以缩放Y轴范围 | Ctrl+滚轮重置缩放"
                self.curve_canvas.create_text(
                    canvas_width // 2, 40,
                    text=hint_text, fill='gray', font=('Arial', 9)
                )

            # 添加ROI尺寸信息
            size_text = f"ROI2: {roi2_width}x{roi2_height}={roi2_width*roi2_height}像素  |  ROI3: {roi3_width}x{roi3_height}={roi3_width*roi3_height}像素"
            self.curve_canvas.create_text(
                canvas_width // 2, canvas_height - 35,
                text=size_text, fill='gray', font=('Arial', 9)
            )

        except Exception as e:
            self.curve_canvas.create_text(
                canvas_width // 2, canvas_height // 2,
                text=f"绘制灰度直方图失败: {str(e)}",
                fill='red', font=('Arial', 10)
            )

    def compute_grayscale_histogram(self, roi1_image, roi_x, roi_y, roi_width, roi_height):
        """计算ROI区域的灰度直方图（0-255灰度值的像素分布）"""
        try:
            # 确保ROI坐标在图片范围内
            img_width, img_height = roi1_image.size
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_x2 = min(img_width, roi_x + roi_width)
            roi_y2 = min(img_height, roi_y + roi_height)

            actual_width = roi_x2 - roi_x
            actual_height = roi_y2 - roi_y

            if actual_width <= 0 or actual_height <= 0:
                return []

            # 裁剪ROI区域
            roi_region = roi1_image.crop((roi_x, roi_y, roi_x2, roi_y2))

            # 转换为灰度图像
            if roi_region.mode != 'L':
                roi_region = roi_region.convert('L')

            # 转换为numpy数组
            import numpy as np
            roi_array = np.array(roi_region)

            # 计算0-255每个灰度值的像素个数
            histogram = [0] * 256
            for gray_val in range(256):
                histogram[gray_val] = int(np.sum(roi_array == gray_val))

            return histogram

        except Exception as e:
            print(f"计算灰度直方图失败: {e}")
            return []

    def compute_roi3_column_mean_gray(self, roi3_image):
        """
        计算ROI3图像每一列的平均灰度值

        Args:
            roi3_image: PIL Image对象（ROI3区域图像）

        Returns:
            list: 每一列的平均灰度值列表（长度=ROI3宽度）
        """
        try:
            import numpy as np

            # 转换为灰度图像
            if roi3_image.mode != 'L':
                roi3_image = roi3_image.convert('L')

            # 转换为numpy数组
            roi3_array = np.array(roi3_image)

            # 计算每一列的平均灰度值（沿垂直方向axis=0计算列均值）
            # roi3_array shape: (height, width)
            # np.mean(axis=0) 对每一列求均值，返回长度为width的数组
            column_means = np.mean(roi3_array, axis=0)

            return column_means.tolist()

        except Exception as e:
            print(f"[ERROR] 计算ROI3列平均灰度值失败: {e}")
            return []

    def on_curve_canvas_mousewheel(self, event):
        """处理曲线画布的鼠标滚轮事件，用于Y轴缩放"""
        try:
            # 获取滚轮滚动方向
            if event.delta:  # Windows
                delta = event.delta / 120  # 标准化为-1或1
            elif event.num == 4:  # Linux 向上滚动
                delta = 1
            elif event.num == 5:  # Linux 向下滚动
                delta = -1
            else:
                return

            # 计算新的缩放因子
            current_zoom = self.roi1_zoom_factor.get()
            if delta > 0:  # 向上滚动，放大
                new_zoom = current_zoom * (1 + self.y_zoom_step)
            else:  # 向下滚动，缩小
                new_zoom = current_zoom * (1 - self.y_zoom_step)

            # 限制缩放范围
            new_zoom = max(self.y_min_zoom, min(self.y_max_zoom, new_zoom))

            # 如果缩放因子发生变化，更新显示
            if abs(new_zoom - current_zoom) > 0.001:
                self.roi1_zoom_factor.set(new_zoom)
                zoom_percentage = int(new_zoom * 100)
                print(f"[INFO] ROI1缩放: {new_zoom:.2f}x ({zoom_percentage}%)")

                # 更新状态栏显示缩放信息
                self.status_var.set(f"ROI1缩放: {zoom_percentage}%")

                # 更新ROI可视化（包括图片和叠加）
                if self.roi1_image:
                    self.update_roi_visualization()
                    # 如果有叠加，也要更新叠加显示
                    if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                        self.update_heat_map_overlay()
                    elif hasattr(self, 'overlay_enabled') and self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                        self.update_overlay()

                # 重新绘制曲线
                roi_config = self.get_roi_config_values()
                original_img_width, original_img_height = self.roi1_image.size
                self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

        except Exception as e:
            print(f"[ERROR] 鼠标滚轮事件处理失败: {e}")

    def on_roi_canvas_mousewheel(self, event):
        """处理ROI1画布的鼠标滚轮事件，用于缩放图片"""
        try:
            # 确保有ROI1图片
            if not self.roi1_image:
                return

            # Linux系统通过event.delta识别滚轮方向
            if hasattr(event, 'delta'):
                # Windows系统: event.delta
                delta = event.delta
            else:
                # Linux系统: event.num属性
                if hasattr(event, 'num'):
                    delta = -event.num  # Linux中num的正负与Windows相反
                else:
                    return  # 如果无法获取滚轮信息，则返回

            # 获取当前ROI1缩放因子
            current_zoom = self.roi1_zoom_factor.get()

            # 根据滚轮方向决定放大或缩小
            if delta > 0:
                # 向上滚轮 - 放大
                new_zoom = current_zoom * (1 + self.y_zoom_step)
            else:
                # 向下滚轮 - 缩小
                new_zoom = current_zoom * (1 - self.y_zoom_step)

            # 限制缩放范围
            new_zoom = max(self.y_min_zoom, min(self.y_max_zoom, new_zoom))

            # 如果缩放因子发生变化，更新显示
            if abs(new_zoom - current_zoom) > 0.001:
                self.roi1_zoom_factor.set(new_zoom)
                zoom_percentage = int(new_zoom * 100)
                print(f"[INFO] ROI1缩放: {new_zoom:.2f}x ({zoom_percentage}%)")

                # 更新状态栏显示缩放信息
                self.status_var.set(f"ROI1缩放: {zoom_percentage}%")

                # 更新ROI可视化（包括图片和叠加）
                if self.roi1_image:
                    self.update_roi_visualization()
                    # 如果有叠加，也要更新叠加显示
                    if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                        self.update_heat_map_overlay()
                    elif hasattr(self, 'overlay_enabled') and self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                        self.update_overlay()

        except Exception as e:
            print(f"[ERROR] ROI1画布鼠标滚轮事件处理失败: {e}")

    def on_roi_canvas_mousewheel_reset(self, event):
        """处理ROI1画布的Ctrl+滚轮重置缩放"""
        try:
            current_zoom = self.roi1_zoom_factor.get()
            if abs(current_zoom - 1.0) > 0.001:
                self.roi1_zoom_factor.set(1.0)
                print(f"[INFO] ROI1缩放已重置: 1.00x")

                # 更新状态栏
                self.status_var.set("ROI1缩放已重置")

                # 更新ROI可视化
                if self.roi1_image:
                    self.update_roi_visualization()
                    # 更新叠加显示
                    if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                        self.update_heat_map_overlay()
                    elif hasattr(self.overlay_enabled) and self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                        self.update_overlay()

        except Exception as e:
            print(f"[ERROR] ROI1画布缩放重置失败: {e}")

    def on_curve_canvas_mousewheel_reset(self, event):
        """重置Y轴缩放为默认值"""
        try:
            if self.y_zoom_factor != 1.0:
                self.y_zoom_factor = 1.0
                print(f"[INFO] Y轴缩放已重置: {self.y_zoom_factor:.2f}x")

                # 重新绘制曲线（如果有ROI图片）
                if self.roi1_image:
                    roi_config = self.get_roi_config_values()
                    original_img_width, original_img_height = self.roi1_image.size
                    self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

        except Exception as e:
            print(f"[ERROR] 重置Y轴缩放失败: {e}")

    def on_roi_canvas_mouse_motion(self, event):
        """处理ROI1画布的鼠标移动事件，显示当前位置的灰度值"""
        try:
            # 确保有ROI1图片
            if not self.roi1_image:
                no_image_msg = "请先导入图片"
                self.pixel_info_var.set(no_image_msg)
                self.status_var.set(no_image_msg)
                return

            # 获取画布尺寸
            canvas_width = self.roi_canvas.winfo_width()
            canvas_height = self.roi_canvas.winfo_height()

            # 如果画布还没有渲染完成，返回
            if canvas_width <= 1 or canvas_height <= 1:
                return

            # 获取原始图片尺寸
            img_width, img_height = self.roi1_image.size

            # 计算图片在画布中的显示区域（考虑缩放和居中）
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            base_scale = min(scale_x, scale_y, 1.0)  # 基础缩放，不放大
            display_scale = base_scale * self.y_zoom_factor  # 应用用户缩放

            # 计算图片在画布中的实际显示区域
            display_width = img_width * display_scale
            display_height = img_height * display_scale
            offset_x = (canvas_width - display_width) / 2
            offset_y = (canvas_height - display_height) / 2

            # 检查鼠标是否在图片区域内
            if (event.x >= offset_x and event.x < offset_x + display_width and
                event.y >= offset_y and event.y < offset_y + display_height):

                # 将画布坐标转换为图片坐标
                img_x = int((event.x - offset_x) / display_scale)
                img_y = int((event.y - offset_y) / display_scale)

                # 确保坐标在图片范围内
                img_x = max(0, min(img_x, img_width - 1))
                img_y = max(0, min(img_y, img_height - 1))

                # 获取该位置的灰度值
                if self.roi1_image.mode == 'RGB':
                    # 如果是RGB图片，转换为灰度值
                    r, g, b = self.roi1_image.getpixel((img_x, img_y))
                    # 使用标准灰度转换公式
                    gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | RGB: ({r:3d}, {g:3d}, {b:3d}) | 灰度: {gray_value:3d}"
                elif self.roi1_image.mode == 'L':
                    # 如果已经是灰度图
                    gray_value = self.roi1_image.getpixel((img_x, img_y))
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | 灰度: {gray_value:3d}"
                else:
                    # 其他模式，转换为RGB再处理
                    rgb_image = self.roi1_image.convert('RGB')
                    r, g, b = rgb_image.getpixel((img_x, img_y))
                    gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | 灰度: {gray_value:3d}"

                # 更新固定文本框显示
                self.pixel_info_var.set(pixel_info)
                # 同时更新状态栏
                self.status_var.set(f"像素信息: {pixel_info}")

            else:
                # 鼠标不在图片区域内，显示提示信息
                canvas_pos_info = f"鼠标位置: ({event.x:3d}, {event.y:3d}) - 在图片区域外"
                self.pixel_info_var.set(canvas_pos_info)
                # 同时更新状态栏
                self.status_var.set(f"画布坐标: ({event.x:3d}, {event.y:3d}) - 在图片区域外")

        except Exception as e:
            print(f"[ERROR] 鼠标移动事件处理失败: {e}")
            error_msg = "获取像素信息失败"
            self.pixel_info_var.set(error_msg)
            self.status_var.set(error_msg)

    def on_roi_canvas_mouse_leave(self, event):
        """处理ROI1画布的鼠标离开事件，重置灰度值显示"""
        try:
            if self.roi1_image:
                leave_msg = "鼠标已离开ROI1区域"
                self.pixel_info_var.set(leave_msg)
                self.status_var.set(leave_msg)
            else:
                no_image_msg = "请导入图片并在ROI1区域移动鼠标"
                self.pixel_info_var.set(no_image_msg)
                self.status_var.set(no_image_msg)
        except Exception as e:
            print(f"[ERROR] 鼠标离开事件处理失败: {e}")

    def on_roi_canvas_pan_start(self, event):
        """处理ROI1画布的拖拽开始事件（鼠标滚轮按下）"""
        try:
            # 确保有ROI1图片
            if not self.roi1_image:
                return

            # 开始拖拽
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_start_offset_x = self.roi1_pan_offset_x.get()
            self.pan_start_offset_y = self.roi1_pan_offset_y.get()

            # 改变鼠标光标
            self.roi_canvas.config(cursor="fleur")

        except Exception as e:
            print(f"[ERROR] 拖拽开始事件处理失败: {e}")

    def on_roi_canvas_pan_motion(self, event):
        """处理ROI1画布的拖拽移动事件（按住滚轮拖拽）"""
        try:
            if not self.is_panning or not self.roi1_image:
                return

            # 计算拖拽偏移量
            delta_x = event.x - self.pan_start_x
            delta_y = event.y - self.pan_start_y

            # 更新平移偏移量
            new_offset_x = self.pan_start_offset_x + delta_x
            new_offset_y = self.pan_start_offset_y + delta_y

            self.roi1_pan_offset_x.set(new_offset_x)
            self.roi1_pan_offset_y.set(new_offset_y)

            # 更新ROI可视化（包括叠加层）
            self.update_roi_visualization()

            # 更新叠加层显示
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                self.update_heat_map_overlay()
            elif self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                self.update_overlay()

            # 显示拖拽信息
            pan_info = f"平移: X={new_offset_x}, Y={new_offset_y}"
            self.status_var.set(pan_info)

        except Exception as e:
            print(f"[ERROR] 拖拽移动事件处理失败: {e}")

    def on_roi_canvas_pan_end(self, event):
        """处理ROI1画布的拖拽结束事件（释放滚轮）"""
        try:
            if not self.is_panning:
                return

            # 结束拖拽
            self.is_panning = False

            # 恢复鼠标光标
            self.roi_canvas.config(cursor="")

            # 显示最终偏移量信息
            final_offset_x = self.roi1_pan_offset_x.get()
            final_offset_y = self.roi1_pan_offset_y.get()

            if final_offset_x != 0 or final_offset_y != 0:
                self.status_var.set(f"平移完成: X={final_offset_x}, Y={final_offset_y}")
            else:
                self.status_var.set("ROI1显示位置已重置")

        except Exception as e:
            print(f"[ERROR] 拖拽结束事件处理失败: {e}")

    def on_roi_canvas_reset_position(self, event):
        """处理ROI1画布双击事件，重置图片位置到居中"""
        try:
            if not self.roi1_image:
                return

            # 重置平移偏移量
            self.roi1_pan_offset_x.set(0)
            self.roi1_pan_offset_y.set(0)

            # 更新ROI可视化（包括叠加层）
            self.update_roi_visualization()

            # 更新叠加层显示
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                self.update_heat_map_overlay()
            elif self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                self.update_overlay()

            # 显示重置信息
            self.status_var.set("ROI1位置已重置到居中")

        except Exception as e:
            print(f"[ERROR] 位置重置事件处理失败: {e}")

    def on_threshold_submit(self):
        """处理阈值提取提交"""
        try:
            if not self.roi1_image:
                self.status_var.set("请先导入ROI1图片")
                return

            # 验证阈值输入
            lower = self.threshold_lower_var.get()
            upper = self.threshold_upper_var.get()

            if lower < 0 or upper > 255 or lower > upper:
                self.status_var.set("阈值范围无效 (0-255, 且下限≤上限)")
                return

            # 如果当前处于热力图模式，清除热力图（互斥显示）
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode:
                self.heat_map = None
                self.heatmap_mode = False

            # 提取ROI3
            roi_config = self.get_roi_config_values()
            roi3_image, roi3_coords = self.extract_roi3_from_roi1(self.roi1_image, roi_config)

            if roi3_image is None:
                self.status_var.set("ROI3提取失败")
                return

            # 应用阈值
            self.threshold_mask = self.apply_threshold_extraction(roi3_image, lower, upper)

            if self.threshold_mask is None:
                self.status_var.set("阈值处理失败")
                return

            # 创建叠加
            self.current_roi3_coords = roi3_coords
            self.current_mask_for_overlay = self.threshold_mask
            self.update_overlay()

            # 计算并显示统计信息
            self.calculate_mask_statistics()

            # 计算阈值范围内像素百分比
            try:
                import numpy as np
                # 转换为灰度图（避免RGB三通道导致重复计算）
                roi3_gray = roi3_image.convert('L')
                roi3_array = np.array(roi3_gray)
                total_roi3_pixels = roi3_array.shape[0] * roi3_array.shape[1]
                in_range_pixels = np.sum((roi3_array >= lower) & (roi3_array <= upper))
                percentage = (in_range_pixels / total_roi3_pixels * 100) if total_roi3_pixels > 0 else 0.0
                self.threshold_percentage_var.set(f"{percentage:.2f}%")
                print(f"[DEBUG] ROI3统计: 总像素={total_roi3_pixels}, 范围[{lower}-{upper}]内像素={in_range_pixels}, 占比={percentage:.2f}%")

                # 计算当前帧的平均灰度值
                current_avg_gray = float(np.mean(roi3_array))

                # 计算与上一帧的差值
                if self.prev_frame_avg_gray is not None:
                    frame_diff = current_avg_gray - self.prev_frame_avg_gray
                    # 显示差值，保留2位小数，正数前面加+号
                    diff_str = f"{frame_diff:+.2f}"
                    self.frame_diff_var.set(diff_str)
                    print(f"[DEBUG] 帧差值: 当前={current_avg_gray:.2f}, 上一帧={self.prev_frame_avg_gray:.2f}, 差值={diff_str}")
                else:
                    self.frame_diff_var.set("--")
                    print(f"[DEBUG] 帧差值: 当前={current_avg_gray:.2f}, 上一帧=无")

                # 保存当前帧的平均灰度值，供下次计算使用
                self.prev_frame_avg_gray = current_avg_gray

            except Exception as e:
                print(f"[ERROR] 百分比计算失败: {e}")
                self.threshold_percentage_var.set("0.00%")

            self.status_var.set(f"阈值提取完成: {lower}-{upper}")

        except Exception as e:
            self.status_var.set(f"阈值提取失败: {str(e)}")

    def on_threshold_clear(self):
        """处理阈值清除"""
        self.threshold_mask = None
        self.largest_component_mask = None
        self.current_overlay_image = None
        self.current_overlay_photo = None
        self.current_roi3_coords = None
        self.current_mask_for_overlay = None

        # 重置统计信息
        self.total_pixels_var.set("0")
        self.threshold_percentage_var.set("0.00%")
        self.frame_diff_var.set("--")
        self.largest_component_pixels_var.set("0")
        self.component_count_var.set("0")
        self.prev_frame_avg_gray = None  # 重置上一帧灰度值

        # 更新画布（不显示叠加）
        self.update_roi_visualization()

        self.status_var.set("阈值已清除")

    def on_max_component(self):
        """处理最大连通域提取"""
        try:
            if self.threshold_mask is None:
                self.status_var.set("请先进行阈值提取")
                return

            # 如果当前处于热力图模式，清除热力图（互斥显示）
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode:
                self.heat_map = None
                self.heatmap_mode = False

            # 执行连通域分析
            largest_mask, largest_area, component_count = self.analyze_connected_components(self.threshold_mask)

            if largest_mask is None:
                self.status_var.set("未找到连通域")
                return

            self.largest_component_mask = largest_mask

            # 使用最大连通域作为当前mask
            self.current_mask_for_overlay = largest_mask

            # 更新叠加
            self.update_overlay()

            # 更新统计信息
            self.largest_component_pixels_var.set(str(largest_area))
            self.component_count_var.set(str(component_count))

            self.status_var.set(f"最大连通域提取完成: {largest_area}像素")

        except Exception as e:
            self.status_var.set(f"连通域分析失败: {str(e)}")

    def auto_execute_threshold_processing(self):
        """自动执行阈值提取和最大连通域处理"""
        try:
            if not self.continuous_check_enabled.get():
                return False

            if not self.roi1_image:
                return False

            print("[INFO] 连续检查模式：自动执行阈值提取...")

            # 执行阈值提取
            self.on_threshold_submit()

            # 延迟执行最大连通域分析
            self.root.after(100, self._auto_execute_max_component)

            # 延迟执行热力图显示（如果启用连续热力图）
            if self.continuous_heatmap_enabled.get():
                self.root.after(200, self._auto_execute_heat_map)

            return True

        except Exception as e:
            print(f"[ERROR] 连续检查执行失败: {e}")
            self.status_var.set(f"连续检查失败: {str(e)}")
            return False

    def auto_execute_heatmap_processing(self):
        """自动执行热力图处理"""
        try:
            if not self.continuous_heatmap_enabled.get():
                return False

            if not self.roi1_image:
                return False

            print("[INFO] 连续热力图模式：自动执行热力图显示...")

            # 延迟执行热力图显示
            self.root.after(100, self._auto_execute_heat_map)

            return True

        except Exception as e:
            print(f"[ERROR] 连续热力图执行失败: {e}")
            self.status_var.set(f"连续热力图失败: {str(e)}")
            return False

    def _auto_execute_max_component(self):
        """延迟执行最大连通域分析"""
        try:
            if self.threshold_mask is not None:
                print("[INFO] 连续检查模式：自动执行最大连通域分析...")
                self.on_max_component()
                self.status_var.set("连续检查：自动完成提取+最大连通域分析")

        except Exception as e:
            print(f"[ERROR] 连续检查-最大连通域执行失败: {e}")

    def _auto_execute_heat_map(self):
        """延迟执行热力图显示"""
        try:
            # 检查是否需要执行热力图显示
            should_execute = False
            status_message = ""

            if self.continuous_check_enabled.get() and self.roi1_image:
                # 连续检查模式下的热力图执行
                should_execute = True
                status_message = "连续检查：自动完成提取+最大连通域+热力图显示"
                print("[INFO] 连续检查模式：自动执行热力图显示...")
            elif self.continuous_heatmap_enabled.get() and self.roi1_image:
                # 独立的连续热力图模式
                should_execute = True
                status_message = "连续热力图：自动完成热力图显示"
                print("[INFO] 连续热力图模式：自动执行热力图显示...")

            if should_execute:
                self.on_heatmap_submit()
                self.status_var.set(status_message)

        except Exception as e:
            print(f"[ERROR] 热力图执行失败: {e}")

    def on_statistics(self):
        """处理统计计算"""
        self.calculate_mask_statistics()

    def on_overlay_toggle(self):
        """处理叠加显示切换"""
        self.update_roi_canvas_with_overlay()

    def on_alpha_change(self, value):
        """处理透明度变化"""
        try:
            alpha = float(value)
            self.alpha_value_label.config(text=f"{alpha:.2f}")
            self.update_overlay()
        except:
            pass

    def on_heatmap_submit(self):
        """处理热力图显示按钮点击"""
        try:
            if not self.roi1_image:
                print("[WARNING] 请先导入ROI1图像")
                return

            self.status_var.set("正在生成热力图...")
            self.root.update()

            # 获取ROI配置
            roi_config = self.get_roi_config_values()
            if not roi_config.get('roi3'):
                print("[WARNING] 请先配置ROI3扩展参数")
                return

            # 提取ROI3图像
            roi3_image, roi3_coords = self.extract_roi3_from_roi1(self.roi1_image, roi_config)
            if roi3_image is None:
                print("[ERROR] ROI3提取失败")
                return

            # 生成热力图
            print("[DEBUG] 开始生成热力图...")
            self.heat_map = self.apply_heat_map_extraction(roi3_image)
            if self.heat_map is None:
                print("[ERROR] 热力图生成失败")
                return

            print(f"[DEBUG] 热力图生成成功: {self.heat_map.shape}")

            # 设置热力图模式
            self.heatmap_mode = True

            # 保存ROI3坐标（用于叠加）
            self.current_roi3_coords = roi3_coords
            print(f"[DEBUG] ROI3坐标已保存: {roi3_coords}")

            # 禁用阈值叠加（互斥显示）
            self.overlay_enabled.set(False)

            # 更新热力图显示
            print("[DEBUG] 开始更新热力图叠加...")
            self.update_heat_map_overlay()
            print("[DEBUG] 热力图叠加更新完成")

            # 更新统计信息
            self.calculate_heat_map_statistics()

            self.status_var.set("热力图显示完成")

        except Exception as e:
            print(f"[ERROR] 热力图处理失败: {e}")
            import traceback
            traceback.print_exc()

    def on_heatmap_clear(self):
        """处理清除热力图按钮点击"""
        try:
            # 清除热力图数据
            self.heat_map = None
            self.heatmap_mode = False

            # 更新画布显示
            self.update_roi_visualization()

            # 清除统计信息
            self.total_pixels_var.set("0")
            self.threshold_percentage_var.set("0.00%")
            self.frame_diff_var.set("--")
            self.largest_pixels_var.set("0")
            self.component_count_var.set("0")
            self.prev_frame_avg_gray = None  # 重置上一帧灰度值

            self.status_var.set("热力图已清除")

        except Exception as e:
            print(f"[ERROR] 清除热力图失败: {e}")

    def on_heatmap_alpha_change(self, value):
        """处理热力图透明度变化"""
        try:
            alpha = float(value)
            self.heatmap_alpha_value_label.config(text=f"{alpha:.2f}")
            if self.heatmap_mode and self.heat_map is not None:
                self.update_heat_map_overlay()
        except:
            pass

    def calculate_heat_map_statistics(self):
        """计算热力图统计信息"""
        try:
            if self.heat_map is not None:
                import numpy as np
                # 统计非零像素（实际上所有像素都有颜色）
                total_pixels = self.heat_map.shape[0] * self.heat_map.shape[1]
                self.total_pixels_var.set(str(total_pixels))

                # 对于热力图，最大连通域就是整个热力图区域
                # 使用getattr避免属性不存在的错误
                if hasattr(self, 'largest_pixels_var'):
                    self.largest_pixels_var.set(str(total_pixels))
                if hasattr(self, 'component_count_var'):
                    self.component_count_var.set("1")
            else:
                self.total_pixels_var.set("0")
                if hasattr(self, 'largest_pixels_var'):
                    self.largest_pixels_var.set("0")
                if hasattr(self, 'component_count_var'):
                    self.component_count_var.set("0")

        except Exception as e:
            print(f"[ERROR] 热力图统计计算失败: {e}")

    def calculate_mask_statistics(self):
        """计算mask统计信息"""
        try:
            if hasattr(self, 'current_mask_for_overlay') and self.current_mask_for_overlay is not None:
                import numpy as np
                mask = self.current_mask_for_overlay
                total_pixels = np.sum(mask > 0)
                self.total_pixels_var.set(str(total_pixels))
            elif self.threshold_mask is not None:
                import numpy as np
                total_pixels = np.sum(self.threshold_mask > 0)
                self.total_pixels_var.set(str(total_pixels))
            else:
                self.total_pixels_var.set("0")

        except Exception as e:
            print(f"[ERROR] 统计计算失败: {e}")

    def update_overlay(self):
        """更新叠加图像"""
        try:
            if not hasattr(self, 'current_roi3_coords') or self.current_roi3_coords is None:
                return

            # 确定使用哪个mask
            if hasattr(self, 'current_mask_for_overlay') and self.current_mask_for_overlay is not None:
                mask = self.current_mask_for_overlay
            elif self.threshold_mask is not None:
                mask = self.threshold_mask
            else:
                return

            # 创建叠加
            overlay = self.create_overlay_image(
                self.roi1_image,
                mask,
                self.current_roi3_coords,
                self.overlay_alpha.get()
            )

            if overlay is not None:
                self.current_overlay_image = overlay
                self.update_roi_canvas_with_overlay()

        except Exception as e:
            print(f"[ERROR] 叠加更新失败: {e}")

    def update_heat_map_overlay(self):
        """更新热力图叠加图像"""
        try:
            if not hasattr(self, 'current_roi3_coords') or self.current_roi3_coords is None:
                return

            if self.heat_map is None:
                return

            # 创建热力图叠加
            overlay = self.create_heat_map_overlay_image(
                self.roi1_image,
                self.heat_map,
                self.current_roi3_coords,
                self.heatmap_alpha_var.get()
            )

            if overlay is not None:
                self.current_overlay_image = overlay
                self.update_roi_canvas_with_overlay()

        except Exception as e:
            print(f"[ERROR] 热力图叠加更新失败: {e}")

    def get_y_zoom_info(self):
        """获取当前Y轴缩放信息"""
        return {
            'zoom_factor': self.y_zoom_factor,
            'zoom_percentage': int(self.y_zoom_factor * 100),
            'is_zoomed': self.y_zoom_factor != 1.0,
            'can_zoom_in': self.y_zoom_factor < self.y_max_zoom,
            'can_zoom_out': self.y_zoom_factor > self.y_min_zoom
        }

    def extract_roi3_from_roi1(self, roi1_image, roi_config):
        """从ROI1图像中提取ROI3区域（优化边界处理）"""
        try:
            if roi1_image is None:
                return None, None

            # 获取ROI1尺寸
            roi1_width, roi1_height = roi1_image.size

            # 计算ROI3区域中心（使用图像中心作为交点）
            center_x = roi1_width // 2
            center_y = roi1_height // 2

            # 从配置中获取ROI3扩展参数
            roi3_params = roi_config['roi3']
            left = roi3_params[0]   # 左扩展
            right = roi3_params[1]  # 右扩展
            top = roi3_params[2]    # 上扩展
            bottom = roi3_params[3]  # 下扩展

            # 计算ROI3区域坐标（不立即裁剪）
            roi3_left = center_x - left
            roi3_top = center_y - top
            roi3_right = center_x + right
            roi3_bottom = center_y + bottom

            # 记录原始计算尺寸（用于调试）
            original_width = roi3_right - roi3_left
            original_height = roi3_bottom - roi3_top
            print(f"[DEBUG] ROI3原始坐标: ({roi3_left}, {roi3_top}, {roi3_right}, {roi3_bottom})")
            print(f"[DEBUG] ROI3原始尺寸: {original_width} x {original_height}")

            # 温和的边界检查：允许边界外扩1-2像素作为缓冲
            buffer = 2  # 2像素缓冲区
            roi3_left = max(-buffer, roi3_left)
            roi3_top = max(-buffer, roi3_top)
            roi3_right = min(roi1_width + buffer, roi3_right)
            roi3_bottom = min(roi1_height + buffer, roi3_bottom)

            # 如果ROI3完全超出ROI1范围，使用中心最小区域
            if roi3_right <= 0 or roi3_left >= roi1_width or roi3_bottom <= 0 or roi3_top >= roi1_height:
                print("[WARNING] ROI3区域完全超出ROI1，使用中心最小区域")
                min_size = 20
                roi3_left = max(0, center_x - min_size // 2)
                roi3_right = min(roi1_width, center_x + min_size // 2)
                roi3_top = max(0, center_y - min_size // 2)
                roi3_bottom = min(roi1_height, center_y + min_size // 2)

            # 最终边界确保（严格，防止越界）
            roi3_left = max(0, roi3_left)
            roi3_top = max(0, roi3_top)
            roi3_right = min(roi1_width, roi3_right)
            roi3_bottom = min(roi1_height, roi3_bottom)

            # 记录实际使用的坐标
            actual_width = roi3_right - roi3_left
            actual_height = roi3_bottom - roi3_top
            print(f"[DEBUG] ROI3实际坐标: ({roi3_left}, {roi3_top}, {roi3_right}, {roi3_bottom})")
            print(f"[DEBUG] ROI3实际尺寸: {actual_width} x {actual_height}")

            # 检查尺寸变化
            if actual_width < original_width or actual_height < original_height:
                print(f"[INFO] ROI3边界被裁剪: 宽度损失{original_width-actual_width}, 高度损失{original_height-actual_height}")

            # 验证ROI3尺寸有效性
            if actual_width <= 0 or actual_height <= 0:
                print(f"[ERROR] 无效的ROI3尺寸: {actual_width} x {actual_height}")
                return None, None

            # 提取ROI3区域
            roi3_region = roi1_image.crop((roi3_left, roi3_top, roi3_right, roi3_bottom))

            return roi3_region, (roi3_left, roi3_top, roi3_right, roi3_bottom)

        except Exception as e:
            print(f"[ERROR] ROI3提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def apply_threshold_extraction(self, roi3_image, lower_thresh, upper_thresh):
        """对ROI3图像应用阈值范围提取，生成二进制mask"""
        try:
            import cv2
            import numpy as np

            if roi3_image is None:
                return None

            # 转换为灰度图（如果还不是）
            if roi3_image.mode != 'L':
                roi3_gray = roi3_image.convert('L')
            else:
                roi3_gray = roi3_image

            # 转换为numpy数组
            roi3_array = np.array(roi3_gray)

            # 应用阈值范围
            mask = cv2.inRange(roi3_array, lower_thresh, upper_thresh)

            return mask

        except Exception as e:
            print(f"[ERROR] 阈值提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def apply_heat_map_extraction(self, roi3_image):
        """对ROI3图像应用热力图提取，生成彩色热力图"""
        try:
            import numpy as np

            if roi3_image is None:
                return None

            # 转换为灰度图（如果还不是）
            if roi3_image.mode != 'L':
                roi3_gray = roi3_image.convert('L')
            else:
                roi3_gray = roi3_image

            # 转换为numpy数组 (0-255范围，uint8类型)
            roi3_array = np.array(roi3_gray, dtype=np.uint8)

            # 尝试使用OpenCV的COLORMAP_JET实现256级连续渐变
            try:
                import cv2
                # COLORMAP_JET: 蓝(低值) → 青绿 → 黄 → 橙 → 红(高值)
                colored_heatmap = cv2.applyColorMap(roi3_array, cv2.COLORMAP_JET)
                # 转换为RGB格式 (OpenCV使用BGR)
                heat_map_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
                return heat_map_rgb

            except ImportError:
                print("[WARNING] OpenCV不可用，使用备用的连续渐变算法")
                # 备用方案：使用numpy插值实现连续渐变
                return self._apply_continuous_heatmap_fallback(roi3_array)

        except Exception as e:
            print(f"[ERROR] 热力图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_continuous_heatmap_fallback(self, roi3_array):
        """使用numpy插值实现连续热力图（备用方案）"""
        try:
            import numpy as np

            # 定义渐变控制点 (蓝 -> 青绿 -> 黄 -> 橙 -> 红)
            # 保持与COLORMAP_JET相似的颜色方向：蓝(低) -> 红(高)
            control_points = {
                0:   [0, 0, 255],      # 纯蓝
                64:  [0, 255, 255],    # 青绿
                128: [255, 255, 0],    # 黄色
                192: [255, 128, 0],    # 橙色
                255: [255, 0, 0]       # 红色
            }

            # 创建连续渐变映射
            heat_map = np.zeros((roi3_array.shape[0], roi3_array.shape[1], 3), dtype=np.uint8)

            # 为每个灰度值计算对应的颜色
            for gray_value in range(256):
                # 找到gray_value所在的区间
                keys = sorted(control_points.keys())
                lower_key = max([k for k in keys if k <= gray_value])
                upper_key = min([k for k in keys if k >= gray_value])

                if lower_key == upper_key:
                    # 正好在控制点上
                    color = control_points[lower_key]
                else:
                    # 线性插值
                    ratio = (gray_value - lower_key) / (upper_key - lower_key)
                    lower_color = np.array(control_points[lower_key])
                    upper_color = np.array(control_points[upper_key])
                    color = lower_color + ratio * (upper_color - lower_color)

                # 应用到对应灰度值的像素
                heat_map[roi3_array == gray_value] = color.astype(np.uint8)

            return heat_map

        except Exception as e:
            print(f"[ERROR] 备用热力图算法失败: {e}")
            return None

    def analyze_connected_components(self, threshold_mask):
        """对阈值mask进行连通域分析，返回最大连通域mask和统计信息"""
        try:
            import cv2
            import numpy as np

            if threshold_mask is None:
                return None, 0, 0

            # 执行连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                threshold_mask, connectivity=8, ltype=cv2.CV_32S
            )

            # 查找最大连通域（排除背景标签0）
            if num_labels > 1:
                largest_area = 0
                largest_label = 1

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > largest_area:
                        largest_area = area
                        largest_label = i

                # 创建仅包含最大连通域的mask
                largest_component_mask = np.zeros_like(threshold_mask)
                largest_component_mask[labels == largest_label] = 255

                total_pixels = np.sum(threshold_mask > 0)
                component_count = num_labels - 1

                return largest_component_mask, largest_area, component_count
            else:
                return None, 0, 0

        except Exception as e:
            print(f"[ERROR] 连通域分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, 0

    def create_overlay_image(self, base_image, mask, roi3_coords, alpha=0.5):
        """创建半透明叠加图像用于画布显示"""
        try:
            import numpy as np
            from PIL import ImageDraw

            if mask is None or roi3_coords is None:
                return None

            # 创建与基础图像相同大小的叠加图像
            overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            roi3_left, roi3_top, roi3_right, roi3_bottom = roi3_coords
            roi3_width = roi3_right - roi3_left
            roi3_height = roi3_bottom - roi3_top

            # 将mask调整为ROI3尺寸
            mask_pil = Image.fromarray(mask)
            mask_resized = mask_pil.resize((roi3_width, roi3_height), Image.Resampling.NEAREST)

            # 创建带透明度的彩色叠加
            overlay_color = (255, 255, 0, int(255 * alpha))  # 黄色带alpha

            # 在mask激活区域应用叠加
            mask_array = np.array(mask_resized)
            for y in range(roi3_height):
                for x in range(roi3_width):
                    if mask_array[y, x] > 0:
                        overlay_draw.point((roi3_left + x, roi3_top + y), fill=overlay_color)

            return overlay

        except Exception as e:
            print(f"[ERROR] 叠加图像创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_heat_map_overlay_image(self, base_image, heat_map, roi3_coords, alpha=0.6):
        """创建热力图叠加图像用于画布显示（优化版）"""
        try:
            import numpy as np
            from PIL import Image, ImageDraw

            if heat_map is None or roi3_coords is None:
                return None

            # 创建与基础图像相同大小的叠加图像
            overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))

            roi3_left, roi3_top, roi3_right, roi3_bottom = roi3_coords
            roi3_width = roi3_right - roi3_left
            roi3_height = roi3_bottom - roi3_top

            # 验证ROI3尺寸
            if roi3_width <= 0 or roi3_height <= 0:
                print(f"[WARNING] 无效的ROI3尺寸: {roi3_width} x {roi3_height}")
                return None

            # 将热力图转换为PIL图像
            if isinstance(heat_map, np.ndarray):
                heat_map_pil = Image.fromarray(heat_map)
            else:
                heat_map_pil = heat_map

            # 根据尺寸差异选择合适的插值方法
            original_size = heat_map_pil.size
            target_size = (roi3_width, roi3_height)

            # 如果尺寸差异较大，使用高质量插值
            size_diff = abs(original_size[0] - roi3_width) + abs(original_size[1] - roi3_height)
            if size_diff > 2:
                resample_method = Image.Resampling.LANCZOS  # 尺寸差异大时使用高质量插值
                print(f"[DEBUG] 热力图缩放: {original_size} -> {target_size} (LANCZOS)")
            else:
                resample_method = Image.Resampling.NEAREST  # 尺寸接近时使用最近邻插值
                print(f"[DEBUG] 热力图缩放: {original_size} -> {target_size} (NEAREST)")

            # 调整热力图尺寸到ROI3大小
            try:
                heat_map_resized = heat_map_pil.resize(target_size, resample_method)
            except Exception as resize_error:
                print(f"[WARNING] 热力图缩放失败，尝试备用方法: {resize_error}")
                # 备用方法：使用简单的resize
                heat_map_resized = heat_map_pil.resize(target_size)

            # 转换为RGBA
            heat_map_rgba = heat_map_resized.convert('RGBA')

            # 创建透明度蒙版（支持渐变透明）
            if isinstance(alpha, (int, float)):
                # 均匀透明度
                alpha_array = np.full((roi3_height, roi3_width), int(255 * alpha), dtype=np.uint8)
                alpha_pil = Image.fromarray(alpha_array)
            elif isinstance(alpha, np.ndarray) and alpha.shape == (roi3_height, roi3_width):
                # 渐变透明度蒙版
                alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8))
            else:
                # 默认60%透明度
                alpha_array = np.full((roi3_height, roi3_width), 153, dtype=np.uint8)
                alpha_pil = Image.fromarray(alpha_array)
                print(f"[WARNING] 无效的alpha值类型 {type(alpha)}，使用默认60%透明度")

            # 将alpha通道应用到热力图
            r, g, b, a = heat_map_rgba.split()
            heat_map_with_alpha = Image.merge('RGBA', (r, g, b, alpha_pil))

            # 将热力图叠加到基础图像上（增强错误处理）
            try:
                overlay.paste(heat_map_with_alpha, (roi3_left, roi3_top), heat_map_with_alpha)
                print(f"[DEBUG] 热力图叠加成功: 位置({roi3_left}, {roi3_top}), 尺寸{target_size}")
            except Exception as paste_error:
                print(f"[WARNING] 热力图叠加失败，尝试边界检查: {paste_error}")
                # 尝试调整粘贴位置以适应边界
                safe_left = max(0, roi3_left)
                safe_top = max(0, roi3_top)
                safe_right = min(base_image.width, roi3_left + roi3_width)
                safe_bottom = min(base_image.height, roi3_top + roi3_height)

                if safe_right > safe_left and safe_bottom > safe_top:
                    crop_width = safe_right - safe_left
                    crop_height = safe_bottom - safe_top
                    cropped_heatmap = heat_map_with_alpha.crop((0, 0, crop_width, crop_height))
                    overlay.paste(cropped_heatmap, (safe_left, safe_top), cropped_heatmap)
                    print(f"[INFO] 热力图已调整并叠加: ({safe_left}, {safe_top}, {safe_right}, {safe_bottom})")
                else:
                    print(f"[ERROR] ROI3区域无法叠加到画布")
                    return None

            return overlay

        except Exception as e:
            print(f"[ERROR] 热力图叠加图像创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_roi_canvas_with_overlay(self):
        """更新ROI1画布叠加显示（支持热力图和阈值mask的互斥显示）"""
        try:
            # 首先调用现有的可视化
            self.update_roi_visualization()

            # 检查是否显示叠加（热力图模式或阈值mask模式）
            show_overlay = False

            # 热力图模式：优先检查是否在热力图模式
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                show_overlay = True
            # 阈值mask模式：检查是否启用阈值叠加
            elif self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                show_overlay = True

            # 如果显示叠加且有叠加图像，则添加到画布
            if show_overlay:
                # 获取画布尺寸
                canvas_width = self.roi_canvas.winfo_width()
                canvas_height = self.roi_canvas.winfo_height()

                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 640
                    canvas_height = 900

                # 计算缩放因子（重用现有逻辑）
                roi_config = self.get_roi_config_values()
                roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']
                roi1_width = roi1_x2 - roi1_x1
                roi1_height = roi1_y2 - roi1_y1

                scale_x = (canvas_width - 20) / roi1_width
                scale_y = (canvas_height - 20) / roi1_height
                base_scale = min(scale_x, scale_y, 1.0)
                scale = base_scale * self.roi1_zoom_factor.get()

                x_offset = (canvas_width - roi1_width * scale) // 2
                y_offset = (canvas_height - roi1_height * scale) // 2

                # 缩放叠加图像以匹配ROI1的缩放
                scaled_overlay = self.current_overlay_image.resize(
                    (int(roi1_width * scale), int(roi1_height * scale)),
                    Image.Resampling.LANCZOS
                )

                # 转换缩放后的叠加图像为PhotoImage
                overlay_photo = ImageTk.PhotoImage(scaled_overlay)

                # 计算叠加层居中位置并应用平移偏移量
                center_x = canvas_width // 2
                center_y = canvas_height // 2

                # 应用与ROI1图像相同的平移偏移量
                pan_offset_x = self.roi1_pan_offset_x.get()
                pan_offset_y = self.roi1_pan_offset_y.get()

                overlay_x = center_x + pan_offset_x
                overlay_y = center_y + pan_offset_y

                # 在画布上绘制叠加（应用平移）
                self.roi_canvas.create_image(
                    overlay_x, overlay_y,
                    image=overlay_photo,
                    anchor=tk.CENTER,
                    tags="overlay"
                )

                # 存储引用以防止垃圾回收
                self.current_overlay_photo = overlay_photo

        except Exception as e:
            print(f"[ERROR] 画布叠加更新失败: {e}")
            import traceback
            traceback.print_exc()

    def _legacy_detect_image_sequence_v0(self, current_file):
        """检测当前图片所在的序列 - 改进的数字索引版本（支持变化时间戳）"""
        try:
            import glob

            # 获取当前文件的目录和文件名
            dir_path = os.path.dirname(current_file)
            filename = os.path.basename(current_file)

            print(f"[DEBUG] 分析文件名: {filename}")

            # 提取数字序列 - 优先提取最长的数字序列
            import re

            # 找到所有数字序列，选择最长的（避免匹配roi1中的'1'）
            all_matches = re.findall(r'(\d+)', filename)
            if not all_matches:
                print(f"[ERROR] 文件名中未找到数字: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            # 选择最长的数字序列作为帧号
            longest_match = max(all_matches, key=len)
            match = re.search(r'(' + longest_match + ')', filename)
            if not match:
                print(f"[ERROR] 无法重新匹配最长数字序列: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            sequence_number = int(match.group(1))
            seq_len = len(longest_match)
            print(f"[DEBUG] 当前文件序列号: {sequence_number} (长度: {seq_len})")

            # 提取文件扩展名
            file_ext = os.path.splitext(filename)[1]

            # 构建搜索模式：支持变化的时间戳
            # glob只支持*和?通配符，不支持正则表达式，需要构建[0-9][0-9]...模式
            # 例如：[0-9][0-9][0-9][0-9][0-9]_*.png
            digit_pattern = ''.join(['[0-9]'] * seq_len)

            if match.start() == 0:
                # 帧号在开头（如00001_xxx.png）
                search_pattern = f"{digit_pattern}_*{file_ext}"
            else:
                # 帧号不在开头（如prefix_00001_xxx.png）
                pattern_prefix = filename[:match.start()]
                search_pattern = f"{pattern_prefix}*{digit_pattern}_*{file_ext}"

            print(f"[DEBUG] 搜索模式: {search_pattern}")

            # 搜索匹配的文件
            search_path = os.path.join(dir_path, search_pattern)
            found_files = glob.glob(search_path)

            print(f"[DEBUG] 找到 {len(found_files)} 个匹配文件")

            # 如果glob没找到，尝试更宽松的模式
            if len(found_files) <= 1:
                # 尝试：保留帧号前缀，只放宽帧号后的命名内容
                # 兼容 roi1_000000.jpg 和 prefix_00001_xxx.png 两种格式
                pattern_prefix = filename[:match.start()]
                relaxed_pattern = f"{pattern_prefix}{digit_pattern}*{file_ext}"
                search_path = os.path.join(dir_path, relaxed_pattern)
                found_files = glob.glob(search_path)
                print(f"[DEBUG] 宽松模式 {relaxed_pattern} 找到 {len(found_files)} 个文件")

            # 提取并排序文件
            sequence_files = []
            for file_path in found_files:
                file_name = os.path.basename(file_path)
                # 在相同位置查找相同长度的数字
                file_match = re.search(r'(\d{' + str(seq_len) + r'})', file_name[match.start():])
                if file_match:
                    file_number = int(file_match.group())
                    sequence_files.append((file_number, file_path))

            # 按数字大小排序
            sequence_files.sort(key=lambda x: x[0])
            sorted_sequence = [path for _, path in sequence_files]

            # 找到当前文件的位置 - 处理路径差异
            try:
                # 标准化路径进行比较
                normalized_current = os.path.normpath(current_file)
                normalized_sequence = [os.path.normpath(path) for path in sorted_sequence]

                current_index = normalized_sequence.index(normalized_current)
                print(f"[SUCCESS] 序列长度: {len(sorted_sequence)}, 当前索引: {current_index}")

                self.current_image_sequence = sorted_sequence
                self.current_image_index = current_index

                print(f"[DEBUG] 检测到完整序列:")
                for i, file_path in enumerate(sorted_sequence):
                    print(f"[DEBUG] {i:2d}: {os.path.basename(file_path)}")
                print(f"[INFO] 当前图片: {os.path.basename(current_file)} (索引 {current_index})")

            except ValueError:
                print(f"[ERROR] 当前文件不在序列中: {filename}")
                print(f"[DEBUG] 当前文件: {os.path.normpath(current_file)}")
                print(f"[DEBUG] 序列文件: {[os.path.basename(p) for p in sorted_sequence[:5]]}...")
                self.current_image_sequence = []
                self.current_image_index = -1

        except Exception as e:
            print(f"[ERROR] 序列检测失败: {e}")
            import traceback
            traceback.print_exc()
            self.current_image_sequence = []
            self.current_image_index = -1

    def _legacy_detect_image_sequence_v1(self, current_file):
        try:
            filename = os.path.basename(current_file)
            dir_path = os.path.dirname(current_file)
            normalized_current = os.path.normpath(current_file)

            print(f"[DEBUG] 鍒嗘瀽鏂囦欢鍚? {filename}")

            numeric_matches = list(re.finditer(r'(\d+)', filename))
            if not numeric_matches:
                print(f"[ERROR] 鏂囦欢鍚嶄腑鏈壘鍒版暟瀛? {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            match = max(numeric_matches, key=lambda item: len(item.group(1)))
            sequence_number = int(match.group(1))
            prefix = filename[:match.start()]
            suffix = filename[match.end():]
            print(f"[DEBUG] 褰撳墠鏂囦欢搴忓垪鍙? {sequence_number}")
            print(f"[DEBUG] 鍖归厤妯℃澘: prefix={prefix!r}, suffix={suffix!r}")

            sequence_pattern = re.compile(
                r'^' + re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$',
                re.IGNORECASE
            )

            sequence_files = []
            for file_path in self.get_supported_image_files_in_folder(dir_path):
                file_name = os.path.basename(file_path)
                file_match = sequence_pattern.match(file_name)
                if not file_match:
                    continue

                file_number = int(file_match.group(1))
                sequence_files.append((file_number, natural_sort_key(file_name), file_path))

            sequence_files.sort(key=lambda item: (item[0], item[1]))
            sorted_sequence = [path for _, _, path in sequence_files]
            normalized_sequence = [os.path.normpath(path) for path in sorted_sequence]

            current_index = normalized_sequence.index(normalized_current)
            print(f"[SUCCESS] 搴忓垪闀垮害: {len(sorted_sequence)}, 褰撳墠绱㈠紩: {current_index}")

            self.current_image_sequence = sorted_sequence
            self.current_image_index = current_index

            print(f"[DEBUG] 妫€娴嬪埌瀹屾暣搴忓垪:")
            for index, file_path in enumerate(sorted_sequence):
                print(f"[DEBUG] {index:2d}: {os.path.basename(file_path)}")
            print(f"[INFO] 褰撳墠鍥剧墖: {os.path.basename(current_file)} (绱㈠紩 {current_index})")

        except ValueError:
            print(f"[ERROR] 褰撳墠鏂囦欢涓嶅湪搴忓垪涓? {os.path.basename(current_file)}")
            self.current_image_sequence = []
            self.current_image_index = -1
        except Exception as e:
            print(f"[ERROR] 搴忓垪妫€娴嬪け璐? {e}")
            import traceback
            traceback.print_exc()
            self.current_image_sequence = []
            self.current_image_index = -1

    def load_image_by_index(self, index):
        """根据索引加载图片"""
        if not self.current_image_sequence or index < 0 or index >= len(self.current_image_sequence):
            return False

        new_file = self.current_image_sequence[index]
        return self.load_image_from_path(new_file, source_label="ROI1图片已加载", reset_frame_diff=False)

    def _legacy_detect_image_sequence_v2(self, current_file):
        try:
            filename = os.path.basename(current_file)
            dir_path = os.path.dirname(current_file)
            normalized_current = os.path.normpath(current_file)

            print(f"[DEBUG] 鍒嗘瀽鏂囦欢鍚? {filename}")

            folder_files = self.get_supported_image_files_in_folder(dir_path)
            numeric_matches = list(re.finditer(r'(\d+)', filename))
            if not numeric_matches:
                print(f"[ERROR] 鏂囦欢鍚嶄腑鏈壘鍒版暟瀛? {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            candidate_sequences = []
            for match in numeric_matches:
                prefix = filename[:match.start()]
                suffix = filename[match.end():]
                pattern = re.compile(
                    r'^' + re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$',
                    re.IGNORECASE
                )

                matched_files = []
                for file_path in folder_files:
                    file_name = os.path.basename(file_path)
                    file_match = pattern.match(file_name)
                    if not file_match:
                        continue

                    file_number = int(file_match.group(1))
                    matched_files.append((file_number, natural_sort_key(file_name), file_path))

                if not matched_files:
                    continue

                matched_files.sort(key=lambda item: (item[0], item[1]))
                candidate_sequences.append({
                    'count': len(matched_files),
                    'digit_length': len(match.group(1)),
                    'start': match.start(),
                    'prefix': prefix,
                    'suffix': suffix,
                    'current_number': int(match.group(1)),
                    'sorted_sequence': [path for _, _, path in matched_files],
                })

            if not candidate_sequences:
                print(f"[ERROR] 鏈壘鍒颁笌褰撳墠鍛藉悕绛夋ā寮忎竴鑷寸殑鍥剧墖搴忓垪: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            best_candidate = max(
                candidate_sequences,
                key=lambda item: (item['count'], item['digit_length'], item['start'])
            )
            sorted_sequence = best_candidate['sorted_sequence']
            normalized_sequence = [os.path.normpath(path) for path in sorted_sequence]

            if normalized_current not in normalized_sequence:
                print(f"[ERROR] 褰撳墠鏂囦欢涓嶅湪搴忓垪涓? {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            current_index = normalized_sequence.index(normalized_current)
            self.current_image_sequence = sorted_sequence
            self.current_image_index = current_index

            print(
                f"[SUCCESS] 搴忓垪闀垮害: {len(sorted_sequence)}, "
                f"褰撳墠绱㈠紩: {current_index}, "
                f"prefix={best_candidate['prefix']!r}, suffix={best_candidate['suffix']!r}, "
                f"current_number={best_candidate['current_number']}"
            )
            print(f"[DEBUG] 妫€娴嬪埌瀹屾暣搴忓垪:")
            for index, file_path in enumerate(sorted_sequence):
                print(f"[DEBUG] {index:2d}: {os.path.basename(file_path)}")
            print(f"[INFO] 褰撳墠鍥剧墖: {os.path.basename(current_file)} (绱㈠紩 {current_index})")

        except Exception as e:
            print(f"[ERROR] 搴忓垪妫€娴嬪け璐? {e}")
            import traceback
            traceback.print_exc()
            self.current_image_sequence = []
            self.current_image_index = -1

    def detect_image_sequence(self, current_file):
        try:
            filename = os.path.basename(current_file)
            dir_path = os.path.dirname(current_file)
            normalized_current = os.path.normpath(current_file)

            print(f"[DEBUG] analyze filename: {filename}")

            folder_files = self.get_supported_image_files_in_folder(dir_path)
            numeric_matches = list(re.finditer(r'(\d+)', filename))
            if not numeric_matches:
                print(f"[ERROR] no numeric token found in filename: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            candidate_sequences = []
            for match in numeric_matches:
                prefix = filename[:match.start()]
                suffix = filename[match.end():]
                pattern = re.compile(
                    r'^' + re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$',
                    re.IGNORECASE
                )

                matched_files = []
                for file_path in folder_files:
                    file_name = os.path.basename(file_path)
                    file_match = pattern.match(file_name)
                    if not file_match:
                        continue

                    file_number = int(file_match.group(1))
                    matched_files.append((file_number, natural_sort_key(file_name), file_path))

                if not matched_files:
                    continue

                matched_files.sort(key=lambda item: (item[0], item[1]))
                candidate_sequences.append({
                    'count': len(matched_files),
                    'digit_length': len(match.group(1)),
                    'start': match.start(),
                    'prefix': prefix,
                    'suffix': suffix,
                    'current_number': int(match.group(1)),
                    'sorted_sequence': [path for _, _, path in matched_files],
                })

            if not candidate_sequences:
                print(f"[ERROR] no matching image sequence found for: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            best_candidate = max(
                candidate_sequences,
                key=lambda item: (item['count'], item['digit_length'], item['start'])
            )
            sorted_sequence = best_candidate['sorted_sequence']
            normalized_sequence = [os.path.normpath(path) for path in sorted_sequence]

            if normalized_current not in normalized_sequence:
                print(f"[ERROR] current file is not in the detected sequence: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            current_index = normalized_sequence.index(normalized_current)
            self.current_image_sequence = sorted_sequence
            self.current_image_index = current_index

            print(
                f"[SUCCESS] sequence length={len(sorted_sequence)}, "
                f"current_index={current_index}, "
                f"prefix={best_candidate['prefix']!r}, suffix={best_candidate['suffix']!r}, "
                f"current_number={best_candidate['current_number']}"
            )
            print("[DEBUG] detected sequence:")
            for index, file_path in enumerate(sorted_sequence):
                print(f"[DEBUG] {index:2d}: {os.path.basename(file_path)}")
            print(f"[INFO] current image: {os.path.basename(current_file)} (index {current_index})")

        except Exception as e:
            print(f"[ERROR] detect sequence failed: {e}")
            import traceback
            traceback.print_exc()
            self.current_image_sequence = []
            self.current_image_index = -1

    def on_key_press(self, event):
        """处理键盘按键事件"""
        try:
            key = event.keysym.lower()
            print(f"[DEBUG] 按键事件: {key}, 序列长度: {len(self.current_image_sequence)}, 当前索引: {self.current_image_index}")

            # 调试：显示所有按键（临时）
            if key not in ['d', 'right', 'a', 'left']:
                print(f"[DEBUG] 忽略按键: {key}")
                return

            # 只在有图片序列时处理导航键
            if not self.current_image_sequence or len(self.current_image_sequence) <= 1:
                print(f"[DEBUG] 无有效图片序列，忽略按键: {key}")
                self.status_var.set("没有检测到图片序列，无法使用键盘导航")
                return

            # 处理导航按键
            if key in ['d', 'right']:  # D 或 → : 下一张图片
                print(f"[DEBUG] 尝试加载下一张图片，当前索引: {self.current_image_index}")
                if self.current_image_index < len(self.current_image_sequence) - 1:
                    success = self.load_image_by_index(self.current_image_index + 1)
                    print(f"[DEBUG] 下一张图片加载结果: {success}")
                else:
                    self.status_var.set("已经是最后一张图片")
                    print(f"[DEBUG] 已经是最后一张图片")

            elif key in ['a', 'left']:  # A 或 ← : 上一张图片
                print(f"[DEBUG] 尝试加载上一张图片，当前索引: {self.current_image_index}")
                if self.current_image_index > 0:
                    success = self.load_image_by_index(self.current_image_index - 1)
                    print(f"[DEBUG] 上一张图片加载结果: {success}")
                else:
                    self.status_var.set("已经是第一张图片")
                    print(f"[DEBUG] 已经是第一张图片")

        except Exception as e:
            print(f"[ERROR] 处理键盘事件失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    root = tk.Tk()

    # 设置窗口标题
    root.title("SimpleFEM 配置管理器")

    # 自动全屏
    root.state('zoomed')  # Windows全屏
    # 如果在其他平台上，可以使用以下方式：
    # root.attributes('-zoomed', True)  # 跨平台全屏

    print("[INFO] 程序已启动并自动全屏")
    print("[INFO] 按 F11 键可以切换全屏模式")

    app = SimpleFEMConfigGUI(root)

    # 设置窗口图标（如果有的话）
    # root.iconbitmap('icon.ico')

    # 添加F11键切换全屏功能
    def toggle_fullscreen(event=None):
        current_state = root.state()
        if current_state == 'zoomed':
            root.state('normal')  # 退出全屏
            print("[INFO] 已退出全屏模式")
        else:
            root.state('zoomed')  # 进入全屏
            print("[INFO] 已进入全屏模式")

    # 绑定F11键切换全屏
    root.bind('<F11>', toggle_fullscreen)

    # 添加ESC键退出全屏功能
    def exit_fullscreen(event=None):
        if root.state() == 'zoomed':
            root.state('normal')
            print("[INFO] 按ESC键退出全屏模式")

    root.bind('<Escape>', exit_fullscreen)

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()
