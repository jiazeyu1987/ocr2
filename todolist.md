# SimpleFEM 红/绿判定接入（已完成）

修改时间：2026-01-27

1) 新增配置：`peak_detect`、`peak_debug_log`、`offline_tmp_frames`（生产 `D:\software_data\settings` 已同步）  
2) before 首帧做焦点识别（绿线交点）并计算 ROI2/ROI3，本轮固定（失败复用上次成功位置）  
3) after 用固定 ROI2 计算灰度均值差 `after-before`，`>= threshold` 判绿，否则判红  
4) 对比截图流程：首帧保存 before；peak/stop/兜底获得 after；可选保存 before→after 全帧到 `D:\software_data\tmp`  
5) 输出清理：移除 `pct=...` 红字标注；打包/编译已支持 `simplefem_focus` 相关文件
