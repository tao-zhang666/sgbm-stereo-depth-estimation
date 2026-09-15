# -*- coding: utf-8 -*-
"""
阶段一 · 04 第四步：uniquenessRatio × disp12MaxDiff 完整网格
================================================================================

【实验范围】
    在第三步保留的 A3 / B3 上完整组合测试：
      `uniquenessRatio = 0、2、5`
      `disp12MaxDiff = -1、0、1、2、5、10、128`
    两个基础配置各测试 21 种组合，共得到 42 组结果。

    `-1` 和 `0` 同时纳入实验，用来观察非正值在当前OpenCV版本中的实际行为；
    较大的 `disp12MaxDiff` 用来观察放宽左右一致性检查后的精度和覆盖率。

【配置】基础配置仍是第三步保留的 A3 / B3；斑点过滤固定 (100,2)，mode 固定 SGBM_3WAY
    （与第五步、第六步的最终值一致，保证可比）

【输出】
    results/04_ur_d12_grid/metrics.csv             42 组完整指标
    results/04_ur_d12_grid/per_scene_d1.csv        150 场景 × 42 组
    results/04_ur_d12_grid/grid_heatmaps.png       网格热力图
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.path.dirname(STAGE_DIR)
GROUP_DIR = os.path.join(PROJ_ROOT, "data_split", "tune")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(STAGE_DIR, ".mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402

# ============================== 实验配置 ==============================
ND = 128
SPECKLE = (100, 2)
MODE = cv2.STEREO_SGBM_MODE_SGBM_3WAY
BASES = [("A3", 7, 784, 3136), ("B3", 9, 1296, 5184)]
UR_LIST = [0, 2, 5]
D12_LIST = [-1, 0, 1, 2, 5, 10, 128]   # 含 -1 与 0：用来确认"负值/零是否真的关闭检查"
REPEATS = 3
FRAME = "10"

OUT_DIR = os.path.join(STAGE_DIR, "results", "04_ur_d12_grid")
# ======================================================================


def imread_any(path, flags):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def is_bad(err, gt):
    return (err > 3) & (err / gt > 0.05)


def main():
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(GROUP_DIR, "image_2"))
                    if n.endswith(".png"))
    keys = [(lab, ur, d12) for (lab, _b, _p1, _p2) in BASES
            for ur in UR_LIST for d12 in D12_LIST]
    meta = {(lab, ur, d12): (bs, p1, p2)
            for (lab, bs, p1, p2) in BASES for ur in UR_LIST for d12 in D12_LIST}

    os.makedirs(OUT_DIR, exist_ok=True)
    print("基础配置: " + " | ".join("%s(bs=%d,P1=%d,P2=%d)" % b for b in BASES))
    print("网格: uniquenessRatio=%s × disp12MaxDiff=%s = %d 组"
          % (UR_LIST, D12_LIST, len(keys)))
    print("固定: nd=%d speckle=%s mode=SGBM_3WAY | 场景 %d 个 | 预热 + 重复 %d 次"
          % (ND, SPECKLE, len(scenes), REPEATS))
    print()

    agg = {k: dict(n=0, bad=0, err_sum=0.0, nv=0, badv=0,
                   nc=0, badc=0, valid=0, total=0, times=[]) for k in keys}
    per_scene = {k: {} for k in keys}
    warmed = set()

    for i, scene in enumerate(scenes):
        gl = imread_any(os.path.join(GROUP_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(GROUP_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(GROUP_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        m = gt > 0
        m_crop = m.copy()
        m_crop[:, :ND] = False

        for k in keys:
            lab, ur, d12 = k
            bs, p1, p2 = meta[k]
            p = dict(minDisparity=0, numDisparities=ND, blockSize=bs, P1=p1, P2=p2,
                     disp12MaxDiff=d12, preFilterCap=63, uniquenessRatio=ur,
                     speckleWindowSize=SPECKLE[0], speckleRange=SPECKLE[1], mode=MODE)
            sgbm = cv2.StereoSGBM_create(**p)
            if k not in warmed:
                sgbm.compute(gl, gr)
                warmed.add(k)

            disp, times = None, []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                d = sgbm.compute(gl, gr).astype(np.float32) / 16.0
                times.append(time.perf_counter() - t0)
                if disp is None:
                    disp = d

            ok = disp >= 0
            err = np.abs(disp - gt)
            a = agg[k]

            e = err[m]
            a["n"] += int(m.sum()); a["bad"] += int(is_bad(e, gt[m]).sum())
            a["err_sum"] += float(e.sum())
            per_scene[k][scene] = 100.0 * float(is_bad(e, gt[m]).mean())

            both = m & ok
            a["nv"] += int(both.sum()); a["badv"] += int(is_bad(err[both], gt[both]).sum())

            a["nc"] += int(m_crop.sum())
            a["badc"] += int(is_bad(err[m_crop], gt[m_crop]).sum())

            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)
            a["times"].append(float(np.median(times)))

        if (i + 1) % 25 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    rows = []
    for k in keys:
        lab, ur, d12 = k
        bs, p1, p2 = meta[k]
        a = agg[k]
        d1s = np.array(list(per_scene[k].values()))
        t = float(np.median(a["times"]))
        rows.append(dict(base=lab, ur=ur, d12=d12, bs=bs, p1=p1, p2=p2, scenes=len(scenes),
                         d1_strict=100.0 * a["bad"] / a["n"],
                         d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
                         d1_common=100.0 * a["badc"] / max(a["nc"], 1),
                         epe=a["err_sum"] / a["n"],
                         valid=100.0 * a["valid"] / a["total"],
                         scene_median=float(np.median(d1s)),
                         scene_p90=float(np.percentile(d1s, 90)),
                         scene_worst=float(d1s.max()),
                         t_med=t, fps=1.0 / t))

    cols = [("基础配置", "base"), ("uniquenessRatio", "ur"), ("disp12MaxDiff", "d12"),
            ("blockSize", "bs"), ("P1", "p1"), ("P2", "p2"), ("场景数", "scenes"),
            ("公共区域D1", "d1_common"), ("严格D1", "d1_strict"), ("有效区域D1", "d1_validreg"),
            ("EPE", "epe"), ("有效像素比例", "valid"),
            ("场景D1中位", "scene_median"), ("场景D1_P90", "scene_p90"),
            ("场景D1最差", "scene_worst"), ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow(["%.4f" % r[kk] if isinstance(r[kk], float) else r[kk] for _, kk in cols])

    with open(os.path.join(OUT_DIR, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["%s_ur%d_d12%d" % k for k in keys])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[k][s] for k in keys])

    # ---------------- 控制台：每个基础配置的公共区域 D1 网格 ----------------
    def matrix(lab, key, fmt, title):
        print("\n%s  [%s]" % (title, lab))
        print("%-12s" % "d12 \\ ur" + "".join("%12d" % ur for ur in UR_LIST))
        for d12 in D12_LIST:
            line = "%-12d" % d12
            for ur in UR_LIST:
                r = next(x for x in rows if x["base"] == lab and x["ur"] == ur and x["d12"] == d12)
                line += "  " + fmt % r[key]
            print(line)

    for lab, _b, _p1, _p2 in BASES:
        matrix(lab, "d1_common", "%10.4f", "公共区域 D1")
        matrix(lab, "valid", "%10.4f", "有效像素比例 (%)")

    # 说明：本脚本只做本轮网格实验与绘图，不读取其它步骤的结果文件。
    # （早期版本的旧数据目录已按要求删除，脚本内不再保留相关引用。）

    # ---------------- 画图 ----------------
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    panels = []
    for lab, _b, _p1, _p2 in BASES:
        panels.append((lab, "d1_common", "公共区域 D1 (%)", "%.2f", "YlOrRd"))
    for lab, _b, _p1, _p2 in BASES:
        panels.append((lab, "valid", "有效像素比例 (%)", "%.1f", "YlGn"))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, (lab, key, title, fmt, cmap) in zip(axes, panels):
        data = np.array([[next(x for x in rows if x["base"] == lab and x["ur"] == ur
                               and x["d12"] == d12)[key]
                          for d12 in D12_LIST] for ur in UR_LIST])
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(D12_LIST)), D12_LIST)
        ax.set_yticks(range(len(UR_LIST)), UR_LIST)
        ax.set_xlabel("disp12MaxDiff")
        ax.set_ylabel("uniquenessRatio")
        ax.set_title("%s：%s" % (lab, title), fontsize=11)
        for r_ in range(data.shape[0]):
            for c_ in range(data.shape[1]):
                ax.text(c_, r_, fmt % data[r_, c_], ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("第四步：uniquenessRatio × disp12MaxDiff 网格（选参组 %d 场景）"
                 % len(scenes), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = os.path.join(OUT_DIR, "grid_heatmaps.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print()
    for name in ("metrics.csv", "per_scene_d1.csv", "grid_heatmaps.png"):
        print("输出 %s" % os.path.join(OUT_DIR, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
