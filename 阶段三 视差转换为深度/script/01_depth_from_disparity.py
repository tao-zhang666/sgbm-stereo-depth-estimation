# -*- coding: utf-8 -*-
"""
阶段三 · 01 由视差计算深度与距离
================================================================================

【本阶段要回答的问题】
    视差只说明"同一个物体在左右图里差了多少像素"，不是距离。
    结合相机标定里的焦距和基线，才能得到"这个物体离车多远"：

        Z = f · B / d          f：焦距（像素）  B：基线（米）  d：视差（像素）

    本脚本读取 KITTI 标定文件，用阶段一锁定的 B5 参数算出视差，再转换成深度，
    并提供"点击测距"：把鼠标点在图上，直接读出该点的视差和距离。

【用法】
    # 1) 最常用：把左右图放进脚本旁边的 target 目录，然后直接运行
    python script\\01_depth_from_disparity.py --click
    #    左右图命名：xxx_L.png / xxx_R.png，或 left.png / right.png，或 xxx_2.png / xxx_3.png
    #    标定：同目录放一个含 P_rect_02 的 txt；或用 --f 与 --baseline 直接给
    #    先看看找到了哪几组： 加 --list

    # 2) 用数据集里的场景（训练集 000000~000199；帧 10 有真值，11 没有）
    python script\\01_depth_from_disparity.py --scene 000006
    python script\\01_depth_from_disparity.py --scene 000006 --frame 11

    # 3) 直接指定任意一对左右图 + 标定
    python script\\01_depth_from_disparity.py --image L.png --right R.png --calib calib.txt

    # 4) 指定若干像素，打印它们的视差与距离（不用图形界面）
    python script\\01_depth_from_disparity.py --scene 000006 --points "600,200;900,180"

    # 5) 交互式点击测距（需要能弹出窗口的桌面环境）
    python script\\01_depth_from_disparity.py --scene 000006 --click

【输入图片的前提】
    左右图必须已经做过立体校正（rectified），即同一物体在左右图中只在水平方向有位移。
    KITTI 的图是校正好的；自建双目需要先做 stereoRectify + remap，否则视差与距离都不对。

【视差有效性与可测距的区别】
    · 视差有效性按 `d ≥ 0` 统计（含 `d = 0`）：`minDisparity=0` 时算法给出 `d=0` 也算“有输出”，
      这类像素约占 1.1%～1.4%，与阶段一的“有效像素比例”、阶段二的“有效视差比例”口径一致；
    · 深度换算必须用 `d > 0`，因为 `Z = f·B/d` 在 `d=0` 时无定义。
    脚本运行时会把这两个比例分别打印出来，不要混用。

【输出】（默认写到 results/depth_<场景或文件名>/）
    depth_cm_16bit.png  深度图，单位**厘米**（16 位，上限 655.35 m）
    depth_m.npy         深度图，float32，单位**米**（完整精度，无截断）
    depth_color.png     深度图的伪彩可视化
    panels.png          左图 / 视差 / 深度 / 视差-距离关系 四联图
    samples.csv         每个指定或点击的点：像素坐标、视差、距离、距离不确定度
    accuracy.csv        仅在有点真值时：按距离分层的深度误差统计
    obstacle.csv        前方最近障碍候选：ROI、候选像素数、距离、支撑像素数

【为什么距离要带不确定度】
    对 Z = fB/d 求导：|dZ| = Z²/(f·B) · |dd|。
    也就是"同样 1 像素的视差误差，在远处会放大成很大的距离误差"，
    所以每个点都同时给出"每 1 像素视差误差对应多少米"。
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.path.dirname(STAGE_DIR)
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "sgbm.yaml")
DATA_DIR = os.path.join(PROJ_ROOT, "data_scene_flow")
CALIB_DIR = os.path.join(PROJ_ROOT, "data_scene_flow_calib")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(STAGE_DIR, ".mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402

MODE_MAP = {"SGBM": cv2.STEREO_SGBM_MODE_SGBM,
            "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            "HH": cv2.STEREO_SGBM_MODE_HH}
DIST_BANDS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)]


def _mini_yaml(path):
    data, cur = {}, None
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace() and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
            data[cur] = {}
            continue
        if ":" in line and cur is not None:
            k, v = line.strip().split(":", 1)
            v = v.strip()
            if v.lstrip("-").isdigit():
                v = int(v)
            data[cur][k] = v
    return data


def load_sgbm_params():
    """从 configs/sgbm.yaml 读锁定参数（有 pyyaml 就用，没有就用极简解析）。"""
    try:
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        cfg = _mini_yaml(CONFIG_PATH)
    p = dict(cfg["sgbm"])
    p["mode"] = MODE_MAP[p["mode"]]
    return p


def imread_any(path, flags):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite_any(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def read_calib(path):
    """读 KITTI calib_cam_to_cam 文件，取 cam2/cam3 的投影矩阵。

    返回 f（像素）、cx、cy、baseline（米）、以及 P2/P3。
    基线来自 P 矩阵第 4 列：P = K·[I|t]，t_x = P[0,3]/f，
    基线 = |t_x(cam3) − t_x(cam2)|。
    """
    vals = {}
    for line in open(path, encoding="utf-8"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        vals[k.strip()] = v.split()
    P2 = np.array([float(x) for x in vals["P_rect_02"]]).reshape(3, 4)
    P3 = np.array([float(x) for x in vals["P_rect_03"]]).reshape(3, 4)
    f = float(P2[0, 0])
    cx, cy = float(P2[0, 2]), float(P2[1, 2])
    t2x, t3x = P2[0, 3] / f, P3[0, 3] / f
    baseline = abs(t3x - t2x)
    return dict(f=f, cx=cx, cy=cy, baseline=baseline, P2=P2, P3=P3)


def disparity_of(gl, gr, params):
    return cv2.StereoSGBM_create(**params).compute(gl, gr).astype(np.float32) / 16.0


def depth_of(disp, f, baseline, max_depth):
    z = np.full(disp.shape, np.nan, dtype=np.float32)
    ok = disp > 0
    z[ok] = f * baseline / disp[ok]
    z[z > max_depth] = np.nan          # 过滤过远/异常值
    return z


def depth_scale(z, max_depth):
    """深度可视化的对数映射：近处 0、远处 1。

    道路场景里大部分像素集中在 3~15 m，直接按 Z 线性上色会把色标挤在一端。
    对数映射能让近处和远处同时有层次；下限取 0.5% 分位数，避免个别极近点拉偏。
    """
    ok = np.isfinite(z)
    z0 = max(float(np.percentile(z[ok], 0.5)) if ok.any() else 1.0, 0.5)
    t = np.zeros(z.shape, dtype=np.float32)
    t[ok] = (np.log(np.clip(z[ok], z0, max_depth)) - np.log(z0)) / (np.log(max_depth) - np.log(z0))
    return np.clip(t, 0, 1), z0


def depth_color(z, max_depth):
    """伪彩深度图：近=红，远=蓝（turbo_r），与四联图里的色标一致。"""
    vis = np.zeros((*z.shape, 3), dtype=np.uint8)
    ok = np.isfinite(z)
    if not ok.any():
        return vis
    t, _ = depth_scale(z, max_depth)
    cmap = plt.get_cmap("turbo_r")
    vis[ok] = (np.array(cmap(t[ok]))[:, :3] * 255).astype(np.uint8)
    return vis


def disp_color(disp, vmax):
    norm = np.clip(disp / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb[disp < 0] = (0, 0, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


SEP_W = 8            # 三格之间的分隔条宽度
BAR_H = 64           # 底部信息条高度


def stack_panels(panels):
    """把若干等大图像横向拼接，中间加分隔条。"""
    h, w = panels[0].shape[:2]
    sep = np.full((h, SEP_W, 3), 45, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def render_hits(base, panel_w, hits, cal, max_depth):
    """把所有测距结果画到画面上：三格同时打十字标记 + 底部信息条显示距离。"""
    vis = base.copy()
    h = base.shape[0] - BAR_H
    for i, r in enumerate(hits):
        for pi in range((base.shape[1] + SEP_W) // (panel_w + SEP_W)):
            x = pi * (panel_w + SEP_W) + r["x"]
            if x >= base.shape[1]:
                continue
            cv2.drawMarker(vis, (x, r["y"]), (60, 60, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, str(i + 1), (x + 10, r["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)
        # 距离文字贴在左图那一格的点击点旁边
        if np.isfinite(r["depth_m"]):
            txt = "%.2f m" % r["depth_m"]
        else:
            txt = "no disparity"
        tx, ty = r["x"] + 14, min(max(r["y"] + 22, 20), h - 8)
        cv2.putText(vis, txt, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
        cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2)

    # 底部信息条：最近一次测距结果（大字号，相当于"弹出距离"）
    cv2.rectangle(vis, (0, h), (vis.shape[1], base.shape[0]), (25, 25, 25), -1)
    if hits:
        r = hits[-1]
        if np.isfinite(r["depth_m"]):
            line = "点击 (%d, %d)   视差 %.2f px   距离 %.2f m   每 1px 视差误差 ±%.3f m" % (
                r["x"], r["y"], r["disparity"], r["depth_m"], r["m_per_px"])
        else:
            line = "点击 (%d, %d)   该点没有有效视差，无法测距" % (r["x"], r["y"])
        cv2.putText(vis, line, (12, h + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        gt = r.get("gt_depth_m", float("nan"))
        gt_err = r.get("depth_err_m", float("nan"))
        if np.isfinite(gt):
            if np.isfinite(gt_err):
                gt_txt = "真值距离 %.2f m（误差 %+.2f m）" % (gt, gt_err)
            else:
                gt_txt = "真值距离 %.2f m（该点无有效视差输出）" % gt
            cv2.putText(vis, gt_txt, (vis.shape[1] - 470, h + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
    else:
        cv2.putText(vis, "在任意一格里点击 → 显示该点的视差与距离（q 退出）", (12, h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    # 三格标题
    for pi, name in enumerate(("左图", "视差图（点击测距）", "深度图（对数色标）")):
        x = pi * (panel_w + SEP_W) + 10
        cv2.putText(vis, name, (x + 1, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, name, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def draw_roi_obstacle(img, roi_px, obs, roi_color, obs_color, thick=2):
    """在图上画出 ROI 框和最近障碍标记。"""
    x0, x1, y0, y1 = roi_px
    cv2.rectangle(img, (x0, y0), (x1, y1), roi_color, thick)
    if obs and obs.get("found"):
        cv2.circle(img, (obs["x"], obs["y"]), 13, obs_color, 2)
        cv2.drawMarker(img, (obs["x"], obs["y"]), obs_color, cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, "nearest obstacle candidate %.1f m" % obs["depth_m"],
                    (max(x0 - 60, 10), max(y0 - 12, 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)
    return img


def build_canvas(gl, disp, depth, max_depth, vmax_disp, roi_px=None, obs=None):
    panels = [cv2.cvtColor(gl, cv2.COLOR_GRAY2RGB),
              disp_color(disp, vmax_disp),
              depth_color(depth, max_depth)]
    if roi_px is not None:
        for p in panels:
            # panels 是 RGB，BGR 顺序的颜色要反过来用
            draw_roi_obstacle(p, roi_px, obs, (0, 255, 255), (255, 90, 90))
    body = stack_panels(panels)
    bar = np.full((BAR_H, body.shape[1], 3), 25, np.uint8)
    # 统一转成 BGR，方便 cv2.imshow / imwrite 直接使用
    return cv2.cvtColor(np.vstack([body, bar]), cv2.COLOR_RGB2BGR), panels[0].shape[1]


def parse_roi(s):
    """ROI 用图像宽高的比例表示，例如 "0.35,0.65,0.55,1.0" = 取下方中央区域（本车道前方）。"""
    v = [float(x) for x in s.split(",")]
    if len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v) or v[0] >= v[1] or v[2] >= v[3]:
        raise ValueError("ROI 需要 4 个 0~1 的比例，且 x0<x1、y0<y1")
    return v


def nearest_obstacle(disp, depth, roi, fb, margin=1.05, min_delta=2.0,
                     support_ratio=0.20, min_support=40):
    """在 ROI 内估计"前方最近障碍距离"。

    为什么不直接用 ROI 内的最小深度：
      · 单个错误匹配像素就可能报出一个 2 米的假障碍，必须要求足够多的支撑像素；
      · 路面本身也占满 ROI，必须把"地面"和"比地面更近的物体"分开。

    做法：
      1) 逐行估计该行**路面**的视差：取该行内有效视差的一个低分位数
         （同一行里，路面通常是比较远的那一档，车辆/行人视差更大）；
      2) 障碍候选 = 视差明显大于同行路面视差的像素（默认超出 5%）；
      3) 对障碍候选的视差做 1 像素分箱直方图并平滑，取主峰；
         从视差最大（最近）的一端往回找，第一个计数达到主峰 20% 的分箱，
         就是"最近的显著障碍面"——这一步排除了孤立噪点。

    返回 dict：检出时用 found=True 并给出距离、视差、像素位置、支撑像素数、路面距离等；
    未检出但候选像素足够时返回 found=False；只有 ROI 内有效像素过少时才返回 None。
    """
    h, w = disp.shape
    x0, x1 = int(roi[0] * w), int(roi[1] * w)
    y0, y1 = int(roi[2] * h), int(roi[3] * h)
    sub_d = disp[y0:y1, x0:x1]
    sub_z = depth[y0:y1, x0:x1]
    ok = np.isfinite(sub_z) & (sub_d > 0)
    if ok.sum() < min_support:
        return None

    # 1) 逐行路面视差：该行有效视差的 25% 分位数
    road = np.full(sub_d.shape[0], np.nan, dtype=np.float32)
    for i in range(sub_d.shape[0]):
        vals = sub_d[i][ok[i]]
        if vals.size >= 10:
            road[i] = np.percentile(vals, 25)

    # 2) 障碍候选
    road_map = np.repeat(road[:, None], sub_d.shape[1], axis=1)
    # 同时要求"相对高出 5%"和"绝对高出 2 像素"，避免把路面自身的视差噪声当成障碍
    cand = ok & np.isfinite(road_map) & (sub_d > road_map * margin) & (sub_d - road_map > min_delta)
    n_cand = int(cand.sum())
    # 路面在 ROI 内最近的深度：用逐行路面基线的 90 分位（稳健，不受单行噪声影响）
    road_vals = road[np.isfinite(road)]
    road_near_m = float(fb / np.percentile(road_vals, 90)) if road_vals.size else float("nan")
    if n_cand < min_support:
        return dict(found=False, n_cand=n_cand, road_nearest_m=road_near_m)

    # 3) 直方图 + 平滑 + 从近到远找第一个显著分箱
    d = sub_d[cand]
    dmax = int(np.ceil(d.max()))
    hist, edges = np.histogram(d, bins=max(dmax, 1), range=(0, max(dmax, 1)))
    k = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    hist_s = np.convolve(hist.astype(np.float32), k / k.sum(), mode="same")
    peak = hist_s.max()
    idx = None
    for b in range(len(hist_s) - 1, -1, -1):          # 从视差最大（最近）往远处扫
        if hist_s[b] >= support_ratio * peak:
            idx = b
            break
    if idx is None:
        return dict(found=False, n_cand=n_cand, road_nearest_m=road_near_m)
    d_est = 0.5 * (edges[idx] + edges[idx + 1])
    band = cand & (np.abs(sub_d - d_est) <= 2.0)
    ys, xs = np.nonzero(band)
    yy, xx = int(np.median(ys)) + y0, int(np.median(xs)) + x0
    z_est = float(depth[yy, xx])
    # 障碍所在行的路面距离（±2 行取中位）：说明"障碍比该处的路面近多少"
    rr = road[max(0, int(np.median(ys)) - 2): int(np.median(ys)) + 3]
    rr = rr[np.isfinite(rr)]
    road_row_m = float(fb / np.median(rr)) if rr.size else float("nan")
    return dict(found=True, depth_m=z_est, disparity=float(disp[yy, xx]), x=xx, y=yy,
                n_cand=n_cand, n_support=int(band.sum()),
                road_nearest_m=road_near_m, road_row_m=road_row_m,
                m_per_px=z_est * z_est / fb)


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# 常见的左右图命名后缀（左边 → 右边）
SUFFIX_PAIRS = [("_l", "_r"), ("_left", "_right"), ("-l", "-r"),
                ("_2", "_3"), ("_0", "_1"), ("im0", "im1"), ("_cam2", "_cam3")]


def _list_images(d):
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT))
    except OSError:
        return []


def _find_calib(d):
    """在目录里找一个含 P_rect_02 的 txt，作为标定文件。"""
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.lower().endswith(".txt"):
            p = os.path.join(d, name)
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    if "P_rect_02" in f.read():
                        return p
            except OSError:
                pass
    return None


def _find_gt(d, stems):
    """如果旁边有 disp_noc_0/，就找同名的当真实视差（可选）。

    stems 是若干个候选名字：图片名、去掉 L/R 后缀的名字、左右图名等，
    都匹配不上时，如果 disp_noc_0 里只有一张图，就直接用它。
    """
    gt_dir = os.path.join(d, "disp_noc_0")
    if not os.path.isdir(gt_dir):
        return None
    files = _list_images(gt_dir)
    for s in stems:
        for f in files:
            if os.path.splitext(f)[0].lower() == str(s).lower():
                return os.path.join(gt_dir, f)
    return os.path.join(gt_dir, files[0]) if len(files) == 1 else None


def find_groups(root):
    """扫描文件夹，找出可处理的左右图组。

    支持三种放法：
      1) 子目录里放 KITTI 风格 image_2/ 与 image_3/（同名文件成对）
      2) 同一个目录里放左右图，按常见后缀配对（xxx_L/xxx_R、left/right、xxx_2/xxx_3 ...）
      3) 目录里恰好两张图 → 按文件名顺序当左右图
    标定：优先用同目录里含 P_rect_02 的 txt。
    """
    groups = []
    if not os.path.isdir(root):
        return groups
    dirs = [root] + [os.path.join(root, x) for x in sorted(os.listdir(root))
                     if os.path.isdir(os.path.join(root, x))]
    for d in dirs:
        if os.path.basename(d) in ("image_2", "image_3"):
            continue
        calib = _find_calib(d)
        d2, d3 = os.path.join(d, "image_2"), os.path.join(d, "image_3")
        if os.path.isdir(d2) and os.path.isdir(d3):          # KITTI 风格
            for f in _list_images(d2):
                if os.path.isfile(os.path.join(d3, f)):
                    stem = os.path.splitext(f)[0]
                    groups.append(dict(name="%s/%s" % (os.path.basename(d), stem),
                                       left=os.path.join(d2, f), right=os.path.join(d3, f),
                                       calib=calib, gt=_find_gt(d, [stem]), how="image_2/image_3"))
            continue
        files = _list_images(d)
        stems = {os.path.splitext(f)[0].lower(): os.path.join(d, f) for f in files}
        used = set()
        for sl, sr in SUFFIX_PAIRS:                          # 按命名后缀配对
            for stem, lp in sorted(stems.items()):
                if stem in used or not stem.endswith(sl):
                    continue
                rstem = stem[:-len(sl)] + sr if sl else sr
                if rstem in stems and rstem not in used:
                    used.add(stem)
                    used.add(rstem)
                    lname = os.path.splitext(os.path.basename(lp))[0]
                    rname = os.path.splitext(os.path.basename(stems[rstem]))[0]
                    groups.append(dict(name=lname, left=lp, right=stems[rstem],
                                       calib=calib,
                                       gt=_find_gt(d, [lname, rname, stem[:-len(sl)] if sl else stem]),
                                       how="%s/%s" % (sl, sr)))
        if not groups and len(files) == 2:                   # 兜底：恰好两张
            lname = os.path.splitext(files[0])[0]
            rname = os.path.splitext(files[1])[0]
            groups.append(dict(name=lname, left=os.path.join(d, files[0]),
                               right=os.path.join(d, files[1]), calib=calib,
                               gt=_find_gt(d, [lname, rname]),
                               how="按文件名顺序（未识别命名）"))
    return sorted(groups, key=lambda g: g["name"])


def print_groups(root, groups):
    print("扫描目录：%s" % root)
    if not groups:
        print("  没有找到可处理的左右图。")
        print("  放法一：xxx_L.png 与 xxx_R.png（或 left/right、xxx_2/xxx_3）放同一目录")
        print("  放法二：建 image_2/ 与 image_3/ 两个子目录，放同名文件")
        print("  标定：同目录放一个含 P_rect_02 的 txt；或用 --f 与 --baseline 直接指定")
        return
    print("  找到 %d 组：" % len(groups))
    for i, g in enumerate(groups, 1):
        print("    [%d] %-22s %s / %s%s%s"
              % (i, g["name"], os.path.basename(g["left"]), os.path.basename(g["right"]),
                 "   标定:" + os.path.basename(g["calib"]) if g["calib"] else "",
                 "   （%s）" % g["how"]))
    print("  用 --pair N 选择第 N 组（不带这个参数时默认处理第 1 组）")


def main():
    ap = argparse.ArgumentParser(description="由视差计算深度与距离（B5 锁定参数）")
    ap.add_argument("--dir", help="从文件夹里找左右图（默认用脚本旁边的 target 目录）")
    ap.add_argument("--pair", type=int, default=1, help="处理第几组图片（默认第 1 组）")
    ap.add_argument("--list", action="store_true", help="只列出文件夹里找到的图片组，不处理")
    ap.add_argument("--scene", help="KITTI 场景号，例如 000006（训练集 000000~000199）")
    ap.add_argument("--frame", default="10", choices=["10", "11"], help="帧号，默认 10（只有 10 有真值）")
    ap.add_argument("--split", default="training", choices=["training", "testing"])
    ap.add_argument("--image", help="自定义左图路径（需同时给 --right，以及 --calib 或 --f/--baseline）")
    ap.add_argument("--right", help="自定义右图路径")
    ap.add_argument("--calib", help="KITTI 格式标定文件（含 P_rect_02 / P_rect_03）")
    ap.add_argument("--f", type=float, help="焦距（像素）——没有 KITTI 标定文件时直接给")
    ap.add_argument("--baseline", type=float, help="基线（米）——没有 KITTI 标定文件时直接给")
    ap.add_argument("--points", help='要测距的像素，形如 "600,200;900,180"')
    ap.add_argument("--click", action="store_true", help="打开三格窗口，鼠标点击测距（需桌面环境）")
    ap.add_argument("--simulate", help='模拟点击坐标，形如 "300,330;640,300"，用于无图形界面时生成窗口预览图')
    ap.add_argument("--max-depth", type=float, default=80.0, help="最大深度（米），超过视为无效，默认 80")
    ap.add_argument("--roi", default="0.35,0.65,0.55,1.0",
                    help="前方兴趣区域（宽高比例 x0,x1,y0,y1），默认下方中央")
    ap.add_argument("--no-obstacle", action="store_true", help="不做前方最近障碍估计")
    ap.add_argument("--out", help="输出目录，默认 results/depth_<场景号>")
    args = ap.parse_args()

    params = load_sgbm_params()

    # ---------- 定位输入 ----------
    # 三种来源：--image 指定图片 ＞ --scene 用数据集场景 ＞ 默认扫描 target 文件夹
    if args.image:
        left_p, right_p = args.image, args.right
        calib_p = args.calib
        tag = os.path.splitext(os.path.basename(args.image))[0]
        gt_p = None
        if not right_p:
            print("错误：用 --image 时必须同时给出 --right（右图）")
            return 2
        if not calib_p and not (args.f and args.baseline):
            print("错误：请用 --calib 给 KITTI 格式标定文件，或用 --f 与 --baseline 直接给出焦距和基线")
            return 2
    elif args.scene:
        base = os.path.join(DATA_DIR, args.split)
        left_p = os.path.join(base, "image_2", "%s_%s.png" % (args.scene, args.frame))
        right_p = os.path.join(base, "image_3", "%s_%s.png" % (args.scene, args.frame))
        calib_p = os.path.join(CALIB_DIR, args.split, "calib_cam_to_cam", "%s.txt" % args.scene)
        gt_p = os.path.join(base, "disp_noc_0", "%s_%s.png" % (args.scene, args.frame))
        tag = "%s_%s" % (args.scene, args.frame)
        if not os.path.isfile(gt_p):
            gt_p = None
    else:
        root_dir = args.dir or os.path.join(SCRIPT_DIR, "target")
        groups = find_groups(root_dir)
        if args.list or not groups:
            print_groups(root_dir, groups)
            return 0 if groups else 2
        if not (1 <= args.pair <= len(groups)):
            print("错误：--pair 只能是 1~%d（共 %d 组）" % (len(groups), len(groups)))
            return 2
        g = groups[args.pair - 1]
        left_p, right_p = g["left"], g["right"]
        calib_p, gt_p = g["calib"], g["gt"]
        tag = g["name"].replace("/", "_")
        print("从 %s 处理第 %d/%d 组：%s（左 %s / 右 %s）"
              % (root_dir, args.pair, len(groups), g["name"],
                 os.path.basename(g["left"]), os.path.basename(g["right"])))

    for p in (left_p, right_p):
        if not p or not os.path.isfile(p):
            print("错误：找不到文件 %s" % p)
            return 2

    out_dir = args.out or os.path.join(STAGE_DIR, "results", "depth_%s" % tag)
    os.makedirs(out_dir, exist_ok=True)

    gl = imread_any(left_p, cv2.IMREAD_GRAYSCALE)
    gr = imread_any(right_p, cv2.IMREAD_GRAYSCALE)

    if calib_p and os.path.isfile(calib_p):
        cal = read_calib(calib_p)
        calib_src = calib_p
    else:
        # 图片名像 KITTI 场景号（前 6 位数字）时，自动用数据集的标定
        guess = tag[:6]
        auto = os.path.join(CALIB_DIR, args.split, "calib_cam_to_cam", "%s.txt" % guess)
        if guess.isdigit() and os.path.isfile(auto):
            cal = read_calib(auto)
            calib_src = "%s（按场景号自动匹配）" % auto
        elif args.f and args.baseline:
            cal = dict(f=float(args.f), cx=float("nan"), cy=float("nan"),
                       baseline=float(args.baseline))
            calib_src = "命令行指定：f = %.4f px，B = %.6f m" % (args.f, args.baseline)
        else:
            print("错误：没有标定文件，也没有给出 --f/--baseline。")
            print("      请把含 P_rect_02 的标定 txt 放到图片同目录，"
                  "或用 --f <焦距像素> --baseline <基线米> 指定。")
            return 2

    print("=" * 78)
    print("阶段三 · 由视差计算深度与距离")
    print("=" * 78)
    print("左图   : %s" % left_p)
    print("右图   : %s" % right_p)
    print("标定   : %s" % calib_src)
    print("图像尺寸: %d x %d" % (gl.shape[1], gl.shape[0]))
    print()
    print("---- 相机参数 ----")
    print("  焦距 f          = %.4f px" % cal["f"])
    if np.isfinite(cal["cx"]):
        print("  主点 (cx, cy)   = (%.4f, %.4f) px" % (cal["cx"], cal["cy"]))
    else:
        print("  主点 (cx, cy)   = 未提供（测距只需要 f 和 B）")
    print("  基线 B          = %.6f m" % cal["baseline"])
    print("  深度公式        Z = f·B / d  →  f·B = %.4f" % (cal["f"] * cal["baseline"]))
    print("  最大深度过滤     Z > %.1f m 视为无效" % args.max_depth)
    print()
    print("  几个参考距离（视差 → 距离）：")
    for d in (128, 64, 32, 16, 8, 4):
        z = cal["f"] * cal["baseline"] / d
        print("    视差 %3d px → %6.2f m%s" % (d, z, "（超出 %.0f m，按无效处理）"
                                              % args.max_depth if z > args.max_depth else ""))

    # ---------- 计算 ----------
    disp = disparity_of(gl, gr, params)
    depth = depth_of(disp, cal["f"], cal["baseline"], args.max_depth)
    valid = np.isfinite(depth)
    print()
    print("---- 结果 ----")
    print("  视差有效像素（算法给出视差，含 d=0） %d (%.2f%%)"
          % (int((disp >= 0).sum()), 100 * (disp >= 0).mean()))
    print("  可测距像素（d>0，用于深度换算）      %d (%.2f%%)"
          % (int((disp > 0).sum()), 100 * (disp > 0).mean()))
    print("  深度有效像素（再经最大深度过滤）      %d (%.2f%%)" % (int(valid.sum()), 100 * valid.mean()))
    if valid.any():
        z = depth[valid]
        print("  距离范围       %.2f ~ %.2f m（中位 %.2f m）"
              % (z.min(), z.max(), float(np.median(z))))

    # ---------- 指定点 ----------
    pts = []
    if args.points:
        for chunk in args.points.split(";"):
            if chunk.strip():
                x, y = chunk.split(",")
                pts.append((int(float(x)), int(float(y))))

    samples = []

    def measure(x, y):
        x = int(np.clip(x, 0, gl.shape[1] - 1))
        y = int(np.clip(y, 0, gl.shape[0] - 1))
        d = float(disp[y, x])
        z = float(depth[y, x]) if valid[y, x] else float("nan")
        # 每 1 像素视差误差带来的距离误差
        unc = (z * z) / (cal["f"] * cal["baseline"]) if np.isfinite(z) else float("nan")
        rec = dict(x=x, y=y, disparity=d, depth_m=z, m_per_px=unc)
        if gt_p:
            gt = imread_any(gt_p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            dg = float(gt[y, x])
            rec["gt_disparity"] = dg
            rec["gt_depth_m"] = (cal["f"] * cal["baseline"] / dg) if dg > 0 else float("nan")
            rec["disp_err_px"] = (d - dg) if dg > 0 else float("nan")
            rec["depth_err_m"] = (z - rec["gt_depth_m"]) if (dg > 0 and np.isfinite(z)) else float("nan")
        return rec

    if pts:
        print()
        print("---- 指定点的测距结果 ----")
        print("  %-12s %10s %10s %12s %12s" % ("像素(x,y)", "视差(px)", "距离(m)", "每1px误差(m)", "真值距离(m)"))
        for (x, y) in pts:
            r = measure(x, y)
            samples.append(r)
            print("  %-12s %10.2f %10.2f %12.3f %12s"
                  % ("(%d,%d)" % (r["x"], r["y"]), r["disparity"], r["depth_m"], r["m_per_px"],
                     ("%.2f" % r["gt_depth_m"]) if ("gt_depth_m" in r and np.isfinite(r["gt_depth_m"])) else "—"))

    # ---------- 前方最近障碍 ----------
    fb = cal["f"] * cal["baseline"]
    roi = parse_roi(args.roi)
    x0, x1 = int(roi[0] * gl.shape[1]), int(roi[1] * gl.shape[1])
    y0, y1 = int(roi[2] * gl.shape[0]), int(roi[3] * gl.shape[0])
    roi_px = (x0, x1, y0, y1)
    obs = None if args.no_obstacle else nearest_obstacle(disp, depth, roi, fb)
    if obs is not None:
        print()
        print("---- 前方最近障碍候选（ROI：x %d~%d，y %d~%d）----" % roi_px)
        print("  说明：这是按「逐行路面基线 + 直方图主峰」这个简单规则给出的候选结果，")
        print("        用于快速估计前方最近的凸起，不保证一定是障碍物。")
        print("  候选像素       %d 个（视差比同一行的路面高 5%% 以上、且高 2 px 以上）" % obs["n_cand"])
        if obs["found"]:
            print("  ★ 最近障碍候选距离 %.2f m   （视差 %.2f px，位置 (%d, %d)，支撑像素 %d 个）"
                  % (obs["depth_m"], obs["disparity"], obs["x"], obs["y"], obs["n_support"]))
            print("    每 1px 视差误差 ±%.3f m" % obs["m_per_px"])
            print("    同行路面距离 %.2f m（障碍比该处路面近 %.2f m）"
                  % (obs["road_row_m"], obs["road_row_m"] - obs["depth_m"]))
        else:
            print("  未检出明显障碍；ROI 内路面最近约 %.2f m" % obs["road_nearest_m"])

    # ---------- 交互式点击测距 ----------
    if args.click or args.simulate:
        canvas, panel_w = build_canvas(gl, disp, depth, args.max_depth,
                                       float(disp.max()) if (disp >= 0).any() else 128.0,
                                       roi_px=roi_px, obs=obs)
        hits = []
        win = "click to measure  (q quit / s save)"

        if args.simulate:
            print()
            print("---- 模拟点击（走的是与窗口完全相同的绘制代码，用于无图形界面时检查）----")
            for chunk in args.simulate.split(";"):
                if not chunk.strip():
                    continue
                x, y = [int(float(v)) for v in chunk.split(",")]
                r = measure(x, y)
                hits.append(r)
                samples.append(r)
                print("  点击 (%4d,%4d)  视差 %7.2f px  距离 %7.2f m"
                      % (r["x"], r["y"], r["disparity"], r["depth_m"]))
            preview = render_hits(canvas, panel_w, hits, cal, args.max_depth)
            imwrite_any(os.path.join(out_dir, "interactive_preview.png"), preview)
            print("  窗口画面已保存：%s" % os.path.join(out_dir, "interactive_preview.png"))
        else:
            print()
            print("---- 交互式点击测距 ----")
            print("  窗口里是 左图 / 视差图 / 深度图 三格。点击任意一格：")
            print("    · 三格同时标出该点；")
            print("    · 底部信息条显示 视差、距离、以及每 1 像素视差误差对应多少米；")
            print("    · 有点位真值时还会显示真值距离与误差。")
            print("  q 或 ESC 退出，s 保存当前画面。")

            def on_mouse(event, x, y, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                px = x % (panel_w + SEP_W)          # 还原成原图坐标（三格等大）
                r = measure(px, y)
                hits.append(r)
                samples.append(r)
                print("  点击 (%4d,%4d)  视差 %7.2f px  距离 %7.2f m" %
                      (r["x"], r["y"], r["disparity"], r["depth_m"]))
                cv2.imshow(win, render_hits(canvas, panel_w, hits, cal, args.max_depth))

            try:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                # 三格拼起来很宽，先缩到能放进屏幕的尺寸（OpenCV 会把鼠标坐标自动换算回原图坐标）
                scale = min(1.0, 1600.0 / canvas.shape[1])
                cv2.resizeWindow(win, int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
                cv2.setMouseCallback(win, on_mouse)
                cv2.imshow(win, render_hits(canvas, panel_w, hits, cal, args.max_depth))
                while True:
                    k = cv2.waitKey(20) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord("s"):
                        shot = render_hits(canvas, panel_w, hits, cal, args.max_depth)
                        imwrite_any(os.path.join(out_dir, "click_screenshot.png"), shot)
                        print("  已保存 %s" % os.path.join(out_dir, "click_screenshot.png"))
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                cv2.destroyAllWindows()
            except cv2.error as e:
                print("  打不开窗口（当前环境没有图形界面）：%s" % str(e).split("\n")[0])
                print("  请在带桌面的机器上运行本命令；也可用 --simulate 生成预览图检查绘制效果。")

    # ---------- 真值校验（按距离分层）----------
    acc_rows = []
    if gt_p:
        gt = imread_any(gt_p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        m = (gt > 0) & valid
        if m.any():
            z_est = depth[m]
            z_gt = cal["f"] * cal["baseline"] / gt[m]
            bands = [z_gt < 10, (z_gt >= 10) & (z_gt < 20), (z_gt >= 20) & (z_gt < 30),
                     (z_gt >= 30) & (z_gt < 50), z_gt >= 50]
            print()
            print("---- 深度精度校验（在有真值的位置）----")
            print("  %-14s %9s %10s %10s %10s" % ("真值距离区间", "像素数", "平均|误差|(m)", "中位|误差|(m)", "相对误差"))
            for (lo, hi), sel in zip(DIST_BANDS, bands):
                if not sel.any():
                    continue
                e = np.abs(z_est[sel] - z_gt[sel])
                rel = e / z_gt[sel]
                name = ("%g~%gm" % (lo, hi)) if hi < 1e8 else (">%gm" % lo)
                print("  %-14s %9d %10.3f %10.3f %9.1f%%"
                      % (name, int(sel.sum()), float(e.mean()), float(np.median(e)), 100 * float(rel.mean())))
                acc_rows.append(dict(区间=name, 像素数=int(sel.sum()), 平均绝对误差m=float(e.mean()),
                                     中位绝对误差m=float(np.median(e)), 平均相对误差=100 * float(rel.mean())))
            e_all = np.abs(z_est - z_gt)
            print("  %-14s %9d %10.3f %10.3f %9.1f%%"
                  % ("全部", int(m.sum()), float(e_all.mean()), float(np.median(e_all)),
                     100 * float((e_all / z_gt).mean())))
            acc_rows.append(dict(区间="全部", 像素数=int(m.sum()), 平均绝对误差m=float(e_all.mean()),
                                 中位绝对误差m=float(np.median(e_all)),
                                 平均相对误差=100 * float((e_all / z_gt).mean())))

    # ---------- 保存 ----------
    # 16 位 PNG 按厘米存：上限 655.35 m，能够覆盖本项目 80 m 的深度上限。
    depth_cm = np.where(valid, np.clip(depth * 100.0, 0, 65535), 0).astype(np.uint16)
    n_clip = int(((depth * 100.0) > 65535).sum())
    imwrite_any(os.path.join(out_dir, "depth_cm_16bit.png"), depth_cm)
    # 另外存一份完整精度的浮点深度（米），供后续程序直接读取
    np.save(os.path.join(out_dir, "depth_m.npy"), np.where(valid, depth, np.nan).astype(np.float32))
    print()
    print("  深度图已保存：depth_cm_16bit.png（单位厘米，上限 655.35 m，本次截断 %d 个像素）"
          % n_clip)
    print("                depth_m.npy（float32，单位米，无截断）")
    imwrite_any(os.path.join(out_dir, "depth_color.png"),
                cv2.cvtColor(depth_color(depth, args.max_depth), cv2.COLOR_RGB2BGR))
    if samples:
        with open(os.path.join(out_dir, "samples.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            w.writeheader()
            for r in samples:
                w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in r.items()})
    if acc_rows:
        with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()))
            w.writeheader()
            for r in acc_rows:
                w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- 四联图 ----------
    names = [x.name for x in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    vis_left = cv2.cvtColor(gl, cv2.COLOR_GRAY2RGB)
    if obs is not None:
        draw_roi_obstacle(vis_left, roi_px, obs, (0, 255, 255), (255, 80, 80))
    _t, _z0 = depth_scale(depth, args.max_depth)      # 供深度图对数色标使用
    for r in samples:
        cv2.circle(vis_left, (r["x"], r["y"]), 6, (255, 0, 0), 2)
        cv2.putText(vis_left, "%.1fm" % r["depth_m"], (r["x"] + 8, r["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    fig, axes = plt.subplots(2, 2, figsize=(17, 8))
    axes[0, 0].imshow(vis_left)
    axes[0, 0].set_title("左图（红点 = 测距点）", fontsize=12)
    axes[0, 1].imshow(disp_color(disp, float(disp.max()) if (disp >= 0).any() else 128))
    axes[0, 1].set_title("SGBM 视差（B5 锁定参数）", fontsize=12)
    _cmap = matplotlib.colormaps["turbo_r"].copy()
    _cmap.set_bad("black")          # 无效像素（NaN）画成黑色，与标题说明一致
    im = axes[1, 0].imshow(np.where(np.isfinite(depth), depth, np.nan), cmap=_cmap,
                           norm=matplotlib.colors.LogNorm(vmin=_z0, vmax=args.max_depth))
    axes[1, 0].set_title("深度图（红=近，蓝=远；黑=无效）  对数色标", fontsize=12)
    cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.03)
    cb.set_label("距离 (m)")
    ticks = [t for t in (2, 3, 5, 10, 20, 40, 80, 150) if _z0 <= t <= args.max_depth]
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(["%g" % t for t in ticks])
    cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # 关掉对数小刻度标签

    ax = axes[1, 1]
    # 横轴范围用"最大有效视差"，不是"最大深度"——两者不是一个量纲
    d_max_axis = float(disp[disp >= 0].max()) if (disp >= 0).any() else 128.0
    dd = np.linspace(1, max(1.0, d_max_axis), 400)
    ax.plot(dd, cal["f"] * cal["baseline"] / dd, color="#1f77b4", lw=2, label="Z = f·B / d")
    for r in samples:
        ax.plot(r["disparity"], r["depth_m"], "o", color="#d62728")
        ax.annotate("(%d,%d)" % (r["x"], r["y"]), (r["disparity"], r["depth_m"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("视差 d (px)")
    ax.set_ylabel("距离 Z (m)")
    ax.set_title("视差-距离关系（f = %.1f px, B = %.4f m）" % (cal["f"], cal["baseline"]), fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("阶段三 由视差计算深度与距离 ｜ %s" % tag, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "panels.png"), dpi=120)
    plt.close(fig)

    if obs is not None:
        with open(os.path.join(out_dir, "obstacle.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["ROI_x0", "ROI_x1", "ROI_y0", "ROI_y1", "候选像素数", "是否检出",
                        "最近障碍候选距离m", "视差px", "位置x", "位置y", "支撑像素数",
                        "每1px误差m", "路面最近距离m", "同行路面距离m"])
            w.writerow([roi_px[0], roi_px[1], roi_px[2], roi_px[3], obs["n_cand"],
                        "是" if obs["found"] else "否",
                        "%.4f" % obs.get("depth_m", float("nan")) if obs["found"] else "—",
                        "%.4f" % obs.get("disparity", float("nan")) if obs["found"] else "—",
                        obs.get("x", "—") if obs["found"] else "—",
                        obs.get("y", "—") if obs["found"] else "—",
                        obs.get("n_support", "—") if obs["found"] else "—",
                        "%.4f" % obs.get("m_per_px", float("nan")) if obs["found"] else "—",
                        "%.4f" % obs["road_nearest_m"],
                        "%.4f" % obs["road_row_m"] if obs["found"] else "—"])

    print()
    for name in ("depth_cm_16bit.png", "depth_m.npy", "depth_color.png", "panels.png"):
        print("输出 %s" % os.path.join(out_dir, name))
    if samples:
        print("输出 %s" % os.path.join(out_dir, "samples.csv"))
    if acc_rows:
        print("输出 %s" % os.path.join(out_dir, "accuracy.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
