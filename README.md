# 基于 Python 和 SGBM 的双目视差与深度估计（KITTI验证）

## 项目简介

本项目使用Python和OpenCV实现基于SGBM的双目视差与深度估计，并以KITTI Stereo 2015作为参数选择和量化评测数据集。项目从SGBM参数实验开始，在固定的选参数据上逐步比较参数组合，再使用未参与选参的场景进行独立评测，最后结合相机标定参数把视差转换为深度并实现点击测距。除KITTI场景外，程序也可以处理满足输入条件的自定义双目图像。

项目重点不是只展示一张视觉效果较好的视差图，而是记录参数如何选择、如何使用真值评价，以及最终参数在新场景上的表现。

## 核心功能

- 使用OpenCV SGBM计算校正后双目图像的视差；
- 读取锁定的参数配置，生成视差图及相关可视化结果；
- 从相机标定参数中获取焦距和双目基线，将视差转换为实际深度；
- 保存可供后续程序读取的浮点深度和16位深度数据；
- 支持指定像素测距和鼠标点击测距；
- 支持输入已经完成立体校正的自定义左右图，并通过KITTI格式标定文件或直接输入焦距、基线完成测距。

## 实验流程

1. 在150个选参场景上分析主要参数及参数组合，并使用D1、EPE、有效视差比例、场景分位数和运行速度进行比较；
2. 锁定最终参数B5，在另外50个未参与选参的场景上进行独立评测；
3. 使用锁定参数生成视差，再结合相机标定参数计算深度并实现点击测距。

## 最终参数

阶段一最终选择配置B5：

| 参数 | 取值 |
| --- | ---: |
| `minDisparity` | 0 |
| `numDisparities` | 128 |
| `blockSize` | 9 |
| `P1` | 1296 |
| `P2` | 5184 |
| `disp12MaxDiff` | 128 |
| `preFilterCap` | 63 |
| `uniquenessRatio` | 0 |
| `speckleWindowSize` | 100 |
| `speckleRange` | 2 |
| `mode` | `SGBM_3WAY` |

程序运行时从 [`configs/sgbm.yaml`](configs/sgbm.yaml) 读取这组参数。

## 独立评测结果

最终参数在50个未参与选参的KITTI场景上得到以下结果：

| 指标 | 结果 |
| --- | ---: |
| 公共区域D1 | 7.1300% |
| 全图严格D1 | 12.5064% |
| 有效区域D1 | 6.0592% |
| 含遮挡D1 | 14.0323% |
| 有效区域EPE | 1.4521 px |
| 真值区域有效率 | 93.1370% |
| 单帧中位耗时 | 0.0387 s |
| FPS | 25.84 |

这些结果来自KITTI Stereo 2015训练集的内部固定划分，不是KITTI官方测试服务器成绩。完整口径和失败案例分析见[阶段二独立评测结果分析](阶段二%20锁定参数后的独立评测/说明文档/阶段二独立评测结果分析.md)。

![阶段二误差可视化](阶段二%20锁定参数后的独立评测/results/holdout_eval/error_visualization.png)

## 深度与测距结果

阶段三从 `P_rect_02` 和 `P_rect_03` 中得到焦距和基线，并把有效视差转换为0～80米范围内的深度。演示场景的有效输出区域平均绝对深度误差为0.8855米，中位绝对误差为0.1415米。该误差只统计同时具有真值和有效预测的位置。

![阶段三视差与深度结果](阶段三%20视差转换为深度/results/depth_demo_L/panels.png)

## 环境配置

本项目实测环境为Windows和Python 3.12.10。建议在项目根目录创建独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要依赖为OpenCV、NumPy和Matplotlib，具体版本见 [`requirements.txt`](requirements.txt)。

## 数据准备

数据来自[KITTI Stereo 2015官方网站](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)。本项目需要下载：

- stereo 2015 / flow 2015 / scene flow 2015 data set；
- calibration files。

完整数据集不包含在本仓库中。解压后在项目根目录形成下面的结构：

```text
data_scene_flow/
├─ training/
│  ├─ image_2/
│  ├─ image_3/
│  ├─ disp_noc_0/
│  └─ disp_occ_0/
└─ testing/
   ├─ image_2/
   └─ image_3/

data_scene_flow_calib/
├─ training/calib_cam_to_cam/
└─ testing/calib_cam_to_cam/
```

### 选参组和验收组

阶段一与阶段二还会读取 `data_split/tune` 和 `data_split/holdout`。每个目录中均需包含 `image_2`、`image_3`、`disp_noc_0` 和 `disp_occ_0` 四个子目录，并只放入训练集的 `_10.png` 文件。

划分规则是固定的，可以用脚本复现：

1. 取出训练集全部场景号 `000000` ~ `000199`（按字符串排序）；
2. 用 Python 标准库 `random.Random(42)` 做一次 shuffle；
3. shuffle 之后**前 150 个作为 `tune`（选参组）**，**末尾 50 个作为 `holdout`（验收组）**。

项目统一以上述 `shuffle` 规则和仓库中的 `scene_split.csv` 为准。复现实验时不要自行更换随机种子或划分规则，以免选参组和验收组发生变化。

```powershell
# 实际创建 data_split/（需要先解压完整数据集）
python ".\阶段一 理解并调整 SGBM 参数\script\00_make_split.py"
# 核对已经创建好的划分，不复制或修改文件
python ".\阶段一 理解并调整 SGBM 参数\script\00_make_split.py" --check
```

仓库中提交了划分清单 [`data_split/scene_split.csv`](data_split/scene_split.csv)（200 行，场景号 → 分组），可以据此核对划分是否正确。验收组只用于阶段二，不应根据其结果继续修改参数。

## 运行方法

### 阶段一：参数实验

按顺序运行：

```powershell
python ".\阶段一 理解并调整 SGBM 参数\script\00_make_split.py"
python ".\阶段一 理解并调整 SGBM 参数\script\01_param_phenomenon.py"
python ".\阶段一 理解并调整 SGBM 参数\script\02_joint_test.py"
python ".\阶段一 理解并调整 SGBM 参数\script\03_smoothness_test.py"
python ".\阶段一 理解并调整 SGBM 参数\script\04_ur_d12_grid.py"
python ".\阶段一 理解并调整 SGBM 参数\script\05_speckle_test.py"
python ".\阶段一 理解并调整 SGBM 参数\script\06_mode_test.py"
```

`00_make_split.py` 会根据固定规则创建 `data_split/tune`、`data_split/holdout` 和 `data_split/scene_split.csv`。数据已经划分完成时，可以加上 `--check` 只做完整性核对。第一步脚本顶部的 `SCAN` 可在 `numDisparities` 和 `blockSize` 之间切换，需要分别运行一次。各脚本的候选参数、评测口径和输出文件均写在文件开头。

### 阶段二：独立评测

```powershell
python ".\阶段二 锁定参数后的独立评测\script\01_holdout_eval.py"
```

该脚本固定读取 `configs/sgbm.yaml`，不会在验收过程中重新调参。

### 阶段三：深度计算与点击测距

仓库中的 `.\阶段三 视差转换为深度\script\target` 提供一组演示输入，不准备完整数据集也可以先运行：

```powershell
python ".\阶段三 视差转换为深度\script\01_depth_from_disparity.py" --click
```

使用完整KITTI数据集中的指定场景：

```powershell
python ".\阶段三 视差转换为深度\script\01_depth_from_disparity.py" --scene 000006
```

脚本还支持自定义左右图、标定文件、焦距、基线、测距坐标和最大深度，使用 `--help` 查看参数。自定义左右图必须来自同一套双目相机并提前完成立体校正；用于测距的标定参数也必须与拍摄图像的相机、分辨率和校正结果相匹配。

当前脚本可以直接读取KITTI格式的 `P_rect_02`、`P_rect_03` 投影矩阵，也可以通过命令行直接提供像素焦距和基线。OpenCV YAML/XML等其他标定文件格式需要先转换，或为脚本增加相应的读取方式。

## 项目结构

```text
SGBM/
├─ configs/                         最终SGBM参数
├─ 阶段一 理解并调整 SGBM 参数/     参数实验、结果和分析
├─ 阶段二 锁定参数后的独立评测/     独立评测、结果和分析
├─ 阶段三 视差转换为深度/           深度计算、演示数据和结果
├─ data_split/scene_split.csv       划分清单（选参组150 + 验收组50）
├─ requirements.txt                 Python依赖
├─ README.md                        项目入口说明
└─ .gitignore                       Git忽略规则
```

本地的 `.venv`、PyCharm配置、Matplotlib缓存、完整KITTI数据及 `data_split` 下的图像不提交到仓库（划分清单 `scene_split.csv` 除外）。实验生成的CSV和代表性图片保留，用于支撑参数选择和结果结论；重新运行即可再生成的 `*.npy` 深度数组不提交。

## 已知限制

- SGBM在遮挡边缘、弱纹理、反光和远距离区域容易出现误匹配或空洞；
- 深度误差会在小视差的远距离区域被明显放大；
- 阶段三深度误差只在一个演示场景的有效预测区域统计；
- 前方障碍结果只是基于视差和道路区域规则得到的候选，不是目标检测结果；
- FPS与计算机硬件、OpenCV版本和运行环境有关；
- 项目没有向KITTI官方测试服务器提交结果；
- B5参数是在KITTI场景上选择的，可以作为其他双目图像的初始配置，但不保证适合不同相机基线、图像分辨率和使用环境。

## 数据来源

本项目使用KITTI Stereo 2015数据集。使用数据时请遵守KITTI网站的相关要求，并在研究或报告中引用其对应论文：

> Moritz Menze and Andreas Geiger. Object Scene Flow for Autonomous Vehicles. CVPR 2015.
