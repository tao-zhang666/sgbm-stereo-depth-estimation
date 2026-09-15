# -*- coding: utf-8 -*-
"""
阶段一 · 03 第三步：P1/P2 平滑强度实验
================================================================================

【这一步在流程里的位置】
    《SGBM最终参数选择方法》§6 第三步：在第二步保留的配置上，对 P1/P2 做小范围调整。
    把公式结果（P1=8*bs*bs, P2=32*bs*bs）当作 1.0 倍，再比较 0.5、1.5、2.0、2.5、3.0 倍，共 6 档。

【第二步保留的两组配置】
    配置 A：numDisparities=128, blockSize=7   → 1.0 倍时 P1=392,  P2=1568
    配置 B：numDisparities=128, blockSize=9   → 1.0 倍时 P1=648,  P2=2592

    P1 和 P2 成对按同一倍数变化，所以始终满足 P2 > P1。

【口径】
    与第二步一致，都只在选参组（data_split/tune，150 场景）上统计：
      · 全图严格 D1：全图、真值>0，算法无效按错误计入
      · 有效区域 D1：只在算法也给出视差的像素上统计（仅作参考，不用于跨配置比较）
      · 公共区域 D1：所有配置统一排除左侧相同宽度后再统计 —— **主要排名依据**
        本步所有配置的 numDisparities 都是 128，所以统一排除左侧 128 列，
        每组都在完全相同的像素集合上评估。

【输出】
    results/03_smoothness_test/metrics.csv
    results/03_smoothness_test/per_scene_d1.csv
    results/03_smoothness_test/smoothness_curves.png
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
ND = 128                                    # 第二步保留的搜索范围
CONFIGS = [                                 # (标签, blockSize, 1.0 倍时的 P1, P2)
    ("A (bs=7)", 7, 392, 1568),
    ("B (bs=9)", 9, 648, 2592),
]
MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # P1/P2 成对缩放的倍数
FRAME = "10"
REPEATS = 2

BASE = dict(
    minDisparity=0,
    numDisparities=ND,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
OUT_SUBDIR = "03_smoothness_test"
# =====================================================================


def imread_any(path, flags):
    """读图，支持含中文路径。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def is_bad(err, gt):
    """KITTI 误判据：绝对误差>3px 且 相对误差>5%。"""
    return (err > 3) & (err / gt > 0.05)


def main():
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(GROUP_DIR, "image_2"))
                    if n.endswith(".png"))
    crop = ND                       # 公共区域：统一排除左侧 numDisparities 列
    keys = [(lab, m) for (lab, bs, p1, p2) in CONFIGS for m in MULTIPLIERS]
    params_of = {}
    for lab, bs, p1, p2 in CONFIGS:
        for m in MULTIPLIERS:
            params_of[(lab, m)] = (bs, int(round(p1 * m)), int(round(p2 * m)))

    out_dir = os.path.join(STAGE_DIR, "results", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print("搜索范围 numDisparities=%d | 公共区域统一排除左侧 %d 列" % (ND, crop))
    print("配置 × 倍数 = %d 组 | 场景 %d 个（选参组）" % (len(keys), len(scenes)))
    for lab, bs, p1, p2 in CONFIGS:
        detail = " | ".join("%.1f倍: %d/%d" % (m, round(p1 * m), round(p2 * m)) for m in MULTIPLIERS)
        print("  %-9s blockSize=%d  %s" % (lab, bs, detail))
    print()

    agg = {k: dict(n=0, bad=0, err_sum=0.0, nv=0, badv=0,
                   nc=0, badc=0, errc_sum=0.0,
                   valid=0, total=0, times=[]) for k in keys}
    per_scene = {k: {} for k in keys}

    for i, scene in enumerate(scenes):
        gl = imread_any(os.path.join(GROUP_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(GROUP_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(GROUP_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        m = gt > 0
        m_crop = m.copy()
        m_crop[:, :crop] = False

        for k in keys:
            bs, p1, p2 = params_of[k]
            p = dict(BASE)
            p.update(blockSize=bs, P1=p1, P2=p2)
            sgbm = cv2.StereoSGBM_create(**p)
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

            e = err[m]                                    # 全图严格
            a["n"] += int(m.sum()); a["bad"] += int(is_bad(e, gt[m]).sum())
            a["err_sum"] += float(e.sum())
            per_scene[k][scene] = 100.0 * float(is_bad(e, gt[m]).mean())

            both = m & ok                                 # 有效区域
            ev = err[both]
            a["nv"] += int(both.sum()); a["badv"] += int(is_bad(ev, gt[both]).sum())

            ec = err[m_crop]                              # 公共区域
            a["nc"] += int(m_crop.sum()); a["badc"] += int(is_bad(ec, gt[m_crop]).sum())
            a["errc_sum"] += float(ec.sum())

            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)
            a["times"].append(float(np.median(times)))

        if (i + 1) % 30 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    rows = []
    for k in keys:
        lab, mult = k
        bs, p1, p2 = params_of[k]
        a = agg[k]
        d1s = np.array(list(per_scene[k].values()))
        t = float(np.median(a["times"]))
        rows.append(dict(label=lab, blockSize=bs, mult=mult, p1=p1, p2=p2,
                         ratio=p2 / p1, scenes=len(scenes),
                         d1_strict=100.0 * a["bad"] / a["n"],
                         d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
                         d1_common=100.0 * a["badc"] / max(a["nc"], 1),
                         epe=a["err_sum"] / a["n"],
                         valid=100.0 * a["valid"] / a["total"],
                         scene_mean=float(d1s.mean()), scene_median=float(np.median(d1s)),
                         scene_p90=float(np.percentile(d1s, 90)), scene_worst=float(d1s.max()),
                         t_med=t, fps=1.0 / t))

    cols = [("配置", "label"), ("倍数", "mult"), ("numDisparities", None), ("blockSize", "blockSize"),
            ("P1", "p1"), ("P2", "p2"), ("P2/P1", "ratio"), ("场景数", "scenes"),
            ("严格D1", "d1_strict"), ("有效区域D1", "d1_validreg"), ("公共区域D1", "d1_common"),
            ("EPE", "epe"), ("有效像素比例", "valid"),
            ("场景D1均值", "scene_mean"), ("场景D1中位", "scene_median"),
            ("场景D1_P90", "scene_p90"), ("场景D1最差", "scene_worst"),
            ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            out = []
            for name, key in cols:
                if key is None:
                    out.append(ND)
                elif key == "ratio":
                    out.append("%.2f" % r[key])
                elif isinstance(r[key], float):
                    out.append("%.4f" % r[key])
                else:
                    out.append(r[key])
            w.writerow(out)

    with open(os.path.join(out_dir, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["%s x%.1f" % (lab, m) for (lab, m) in keys])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[k][s] for k in keys])

    # ---------------- 控制台 ----------------
    print()
    print("%-9s %-6s %-6s %-6s %-9s %-9s %-9s %-9s %-8s %-6s"
          % ("配置", "倍数", "P1", "P2", "公共区域D1", "严格D1", "有效区域D1", "有效像素", "FPS", "P90"))
    for r in rows:
        print("%-9s %-6.1f %-6d %-6d %-9.2f %-9.2f %-9.2f %-8.1f%% %-6.1f %-6.2f"
              % (r["label"], r["mult"], r["p1"], r["p2"], r["d1_common"], r["d1_strict"],
                 r["d1_validreg"], r["valid"], r["fps"], r["scene_p90"]))

    # 同一配置内，哪个倍数最好
    print()
    for lab, bs, p1, p2 in CONFIGS:
        sub = [r for r in rows if r["label"] == lab]
        best_c = min(sub, key=lambda r: r["d1_common"])
        best_s = min(sub, key=lambda r: r["d1_strict"])
        print("%s：公共区域 D1 最优 = %.1f 倍（%.2f）；严格 D1 最优 = %.1f 倍（%.2f）"
              % (lab, best_c["mult"], best_c["d1_common"], best_s["mult"], best_s["d1_strict"]))

    # ---------------- 画图 ----------------
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    panels = [("d1_common", "公共区域 D1（主要依据，越低越好）", "%"),
              ("d1_strict", "全图严格 D1（无效算错）", "%"),
              ("valid", "有效像素比例", "%"),
              ("fps", "FPS", "")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, (key, title, unit) in zip(axes, panels):
        for lab, bs, p1, p2 in CONFIGS:
            sub = sorted([r for r in rows if r["label"] == lab], key=lambda r: r["mult"])
            xs = [r["mult"] for r in sub]
            ys = [r[key] for r in sub]
            ax.plot(xs, ys, "o-", label="%s  P1(1.0×)=%d, P2(1.0×)=%d" % (lab, p1, p2))
            for x, y in zip(xs, ys):
                ax.annotate(("%.2f" % y) if key != "fps" else ("%.1f" % y),
                            (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)
        ax.set_xlabel("P1/P2 倍数（1.0 = 公式值）")
        ax.set_ylabel(title + (" (%)" if unit else ""))
        ax.set_xticks(MULTIPLIERS)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("第三步 P1/P2 平滑强度实验（选参组 %d 个场景，numDisparities=%d）"
                 % (len(scenes), ND), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = os.path.join(out_dir, "smoothness_curves.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print()
    print("输出 %s" % os.path.join(out_dir, "metrics.csv"))
    print("输出 %s" % os.path.join(out_dir, "per_scene_d1.csv"))
    print("输出 %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
