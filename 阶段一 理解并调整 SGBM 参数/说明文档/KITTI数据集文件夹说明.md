# KITTI 2015 数据集文件夹说明

## 1. 当前下载情况

项目目录中包含两个 KITTI 文件夹：

```text
SGBM/
├── data_scene_flow/
└── data_scene_flow_calib/
```

- `data_scene_flow`：双目图像、真值视差、光流和辅助标注。
- `data_scene_flow_calib`：相机、激光雷达和 IMU 之间的标定参数。

## 2. 检查结果

当前文件夹结构和文件数量正常，没有发现明显的缺失或错误。

检查结果如下：

| 检查项目 | 结果 |
|---|---|
| 训练集左图数量 | 400 张，正常 |
| 训练集右图数量 | 400 张，正常 |
| 测试集左图数量 | 400 张，正常 |
| 测试集右图数量 | 400 张，正常 |
| 每类训练集视差真值 | 200 张，正常 |
| 每类训练集光流真值 | 200 张，正常 |
| 训练集相机标定文件 | 200 个，正常 |
| 测试集相机标定文件 | 200 个，正常 |
| 左右图像文件名 | 一一对应 |
| 空文件 | 未发现 |
| 标定文件关键字段 | `P_rect_02`、`P_rect_03` 均存在 |

抽查的第一组数据也能被 OpenCV 正常读取：

```text
左图：375 × 1242，3 通道 uint8
右图：375 × 1242，3 通道 uint8
视差真值：375 × 1242，单通道 uint16
```

这些检查说明数据已经正确解压，可以开始项目。

---

# 一、data_scene_flow 文件夹

## 1. 总体结构

```text
data_scene_flow/
├── training/
│   ├── image_2/
│   ├── image_3/
│   ├── disp_noc_0/
│   ├── disp_noc_1/
│   ├── disp_occ_0/
│   ├── disp_occ_1/
│   ├── flow_noc/
│   ├── flow_occ/
│   ├── obj_map/
│   ├── viz_flow_occ/
│   └── viz_flow_occ_dilate_1/
└── testing/
    ├── image_2/
    └── image_3/
```

`training` 是训练和本地评测数据，包含参考答案；`testing` 是官方测试数据，不公开参考答案。

对于本项目，应当先使用 `training`，因为只有训练集能够比较 SGBM 结果和真值视差。

## 2. image_2：左相机彩色图像

```text
data_scene_flow/training/image_2/
data_scene_flow/testing/image_2/
```

`image_2` 保存 KITTI 左侧彩色相机图像。

例如：

```text
000000_10.png
000000_11.png
```

- `000000`：场景编号。
- `_10`：第一个时刻的图像。
- `_11`：下一个时刻的图像。

200 个场景中的每个场景都有 `_10` 和 `_11` 两张图像，因此该文件夹共有 400 张图像。

刚开始学习 SGBM 时，只使用 `_10` 图像即可：

```text
data_scene_flow/training/image_2/000000_10.png
```

## 3. image_3：右相机彩色图像

```text
data_scene_flow/training/image_3/
data_scene_flow/testing/image_3/
```

`image_3` 保存 KITTI 右侧彩色相机图像。

计算双目视差时，左右图像的场景编号和时刻必须完全相同。例如：

```text
左图：training/image_2/000000_10.png
右图：training/image_3/000000_10.png
```

不能将 `000000_10.png` 和 `000001_10.png` 配对，也不能将 `_10` 和 `_11` 配对。

## 4. disp_occ_0：第一个时刻的含遮挡评测视差

```text
data_scene_flow/training/disp_occ_0/
```

该文件夹保存第一个时刻的真值视差，用于包含遮挡情况的评测。

例如：

```text
data_scene_flow/training/disp_occ_0/000000_10.png
```

它对应：

```text
training/image_2/000000_10.png
training/image_3/000000_10.png
```

这是用于**含遮挡**整体评测的真值文件夹。本项目的主要选参真值是 `disp_noc_0`（见下一节），`disp_occ_0` 用于补充统计含遮挡情况的 D1。

需要注意：真值文件虽然是 PNG 图片，但里面保存的是 16 位数值，不能作为普通彩色图片读取。一般使用：

```python
# 注意：本项目路径含中文，cv2.imread 在非 ASCII 路径上会失败（静默返回 None），
#       因此统一使用 np.fromfile + cv2.imdecode 读取
gt_raw = cv2.imdecode(np.fromfile(gt_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
gt_disparity = gt_raw.astype(np.float32) / 256.0
valid_mask = gt_raw > 0
```

## 5. disp_noc_0：第一个时刻的非遮挡视差

```text
data_scene_flow/training/disp_noc_0/
```

`noc` 是 non-occluded 的缩写，表示主要评测没有被遮挡的有效区域。

它和 `disp_occ_0` 的区别可以简单理解为：

- `disp_noc_0`：重点评测非遮挡区域；
- `disp_occ_0`：用于包含遮挡情况的整体评测。

本项目以 `disp_noc_0` 作为主要真值，`disp_occ_0` 只用于补充统计含遮挡口径的 D1（阶段二中用到），不作为选参依据。

## 6. disp_occ_1 和 disp_noc_1：第二个时刻的视差

```text
data_scene_flow/training/disp_occ_1/
data_scene_flow/training/disp_noc_1/
```

这两个文件夹保存第二个时刻的真值视差，对应左右相机的 `_11` 图像。**注意：这两个目录里的文件仍然命名为 `<场景号>_10.png`（`_11` 只出现在 `image_2`/`image_3` 中），不要按文件名去找 `_11`。**

初学阶段只计算 `_10` 图像的视差，因此暂时不需要使用这两个文件夹。

## 7. flow_occ 和 flow_noc：光流真值

```text
data_scene_flow/training/flow_occ/
data_scene_flow/training/flow_noc/
```

光流描述同一相机中，像素从第一个时刻到第二个时刻的运动：

```text
左相机 _10 图像 → 左相机 _11 图像
```

它与双目视差不同：

- 双目视差比较同一时刻的左图和右图；
- 光流比较同一相机在两个时刻的图像。

当前 SGBM 项目不需要光流文件，可以暂时忽略。

## 8. obj_map：前景与背景辅助标注

```text
data_scene_flow/training/obj_map/
```

这是用于区分场景中前景目标和背景区域的辅助标注，可以帮助分别统计车辆等前景区域与背景区域的误差。

当前阶段不需要使用，后续如果要分析“车辆区域的视差误差”，可以再研究。

## 9. viz_flow_occ 和 viz_flow_occ_dilate_1

```text
data_scene_flow/training/viz_flow_occ/
data_scene_flow/training/viz_flow_occ_dilate_1/
```

这些是与光流显示或辅助处理有关的可视化文件，不是 SGBM 计算的必要输入。

当前项目可以完全忽略。

---

# 二、data_scene_flow_calib 文件夹

## 1. 总体结构

```text
data_scene_flow_calib/
├── training/
│   ├── calib_cam_to_cam/
│   ├── calib_velo_to_cam/
│   └── calib_imu_to_velo/
└── testing/
    ├── calib_cam_to_cam/
    ├── calib_velo_to_cam/
    └── calib_imu_to_velo/
```

训练集和测试集各有 200 个场景，因此每个标定子文件夹中有 200 个文本文件：

```text
000000.txt
000001.txt
...
000199.txt
```

标定文件必须和图像场景编号对应。例如：

```text
图像：data_scene_flow/training/image_2/000000_10.png
标定：data_scene_flow_calib/training/calib_cam_to_cam/000000.txt
```

## 2. calib_cam_to_cam：相机标定参数

```text
data_scene_flow_calib/training/calib_cam_to_cam/
data_scene_flow_calib/testing/calib_cam_to_cam/
```

这是当前项目唯一需要使用的标定文件夹。

每个文件中包含多台相机的：

- 原始图像尺寸；
- 相机内参；
- 镜头畸变参数；
- 相机旋转和平移参数；
- 校正矩阵；
- 校正后的投影矩阵。

KITTI 中：

- `_02` 表示左侧彩色相机；
- `_03` 表示右侧彩色相机。

由于 `image_2` 和 `image_3` 已经是校正后的图像，本项目后续主要读取：

```text
P_rect_02
P_rect_03
```

- `P_rect_02`：左侧彩色相机的校正投影矩阵；
- `P_rect_03`：右侧彩色相机的校正投影矩阵。

可以从这两个矩阵中获得计算深度需要的焦距、主点和双目基线。

本数据包中使用的字段名称是 `P_rect_02` 和 `P_rect_03`，不是某些简化教程中写的 `P2` 和 `P3`。编写读取程序时必须使用数据文件中的真实字段名。

## 3. calib_velo_to_cam：激光雷达到相机的标定

```text
data_scene_flow_calib/training/calib_velo_to_cam/
data_scene_flow_calib/testing/calib_velo_to_cam/
```

`velo` 指 KITTI 使用的 Velodyne 激光雷达。

这些文件描述如何把激光雷达坐标转换到相机坐标，主要用于：

- 相机和激光雷达数据融合；
- 将激光雷达点投影到图像；
- 使用激光雷达验证深度。

当前只做 SGBM 双目视差，不需要使用。

## 4. calib_imu_to_velo：IMU 到激光雷达的标定

```text
data_scene_flow_calib/training/calib_imu_to_velo/
data_scene_flow_calib/testing/calib_imu_to_velo/
```

这些文件描述 IMU/GPS 坐标系与激光雷达坐标系之间的变换，主要用于车辆定位、轨迹和多传感器融合。

当前只做 SGBM 双目视差，不需要使用。
