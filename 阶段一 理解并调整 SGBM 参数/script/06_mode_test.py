# -*- coding: utf-8 -*-
"""
阶段一 · 06 第六步：比较运行模式（SGBM / SGBM_3WAY / HH）
================================================================================

【这一步在流程里的位置】
    《SGBM最终参数选择方法》§6 第六步：固定其他参数，比较三种代价聚合模式，
    使用完全相同的数据和指标，比较精度与速度。
    "路径更多不代表一定更适合本项目，应以实际测量为准。"

【第五步保留的两个基础配置】
    A4：numDisparities=128, blockSize=7, P1=784,  P2=3136,
        uniquenessRatio=0, disp12MaxDiff=128, speckleWindowSize=100, speckleRange=2
    B4：同上，blockSize=9, P1=1296, P2=5184

    只改 mode：SGBM（5 方向）/ SGBM_3WAY（3 方向）/ HH（8 方向）
    → 3 × 2 = 6 组完整参数。

【指标】与第五步相同
    · 全图严格 D1 / 有效区域 D1 / 公共区域 D1 / EPE / 有效像素比例 / 场景统计
    · 耗时（预热 + 重复 3 次取中位）、FPS
    · 小斑块数（有效掩膜上面积小于 100 px 的连通块个数）：本步 speckleWindowSize=100
      会删掉所有面积小于 100 的块，因此三种模式该值均为 0，仅作为参考列，不参与选择
【输出】
    results/06_mode_test/metrics.csv
    results/06_mode_test/per_scene_d1.csv
    results/06_mode_test/mode_curves.png      精度 / 覆盖率 / 速度对比图
    results/06_mode_test/sample_maps_A4.png   三种模式的视差图对比
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
ND, UR, D12 = 128, 0, 128
SPECKLE = (100, 2)
BASES = [("A4", 7, 784, 3136), ("B4", 9, 1296, 5184)]
MODES = [("SGBM", cv2.STEREO_SGBM_MODE_SGBM),
         ("SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM_3WAY),
         ("HH", cv2.STEREO_SGBM_MODE_HH)]
REPEATS = 3
FRAME = "10"
VIS_SCENE = "000000"
SPECKLE_PX = 100
OUT_DIR = os.path.join(STAGE_DIR, "results", "06_mode_test")
# ======================================================================


def imread_any(path, flags):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def is_bad(err, gt):
    return (err > 3) & (err / gt > 0.05)


def colorize(disp, vmax):
    norm = np.clip(disp / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb[disp < 0] = (0, 0, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def main():
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(GROUP_DIR, "image_2"))
                    if n.endswith(".png"))
    keys = [(lab, mname) for (lab, _b, _p1, _p2) in BASES for (mname, _m) in MODES]
    meta = {(lab, mname): (bs, p1, p2, mval)
            for (lab, bs, p1, p2) in BASES for (mname, mval) in MODES}

    os.makedirs(OUT_DIR, exist_ok=True)
    print("基础配置: " + " | ".join("%s(bs=%d,P1=%d,P2=%d)" % b for b in BASES))
    print("固定: nd=%d ur=%d d12=%d speckle=%s" % (ND, UR, D12, SPECKLE))
    print("比较模式: %s → %d 组 | 场景 %d 个" % ([m[0] for m in MODES], len(keys), len(scenes)))
    print()

    agg = {k: dict(n=0, bad=0, err_sum=0.0, nv=0, badv=0,
                   nc=0, badc=0, valid=0, total=0, times=[],
                   comp_small=0) for k in keys}
    per_scene = {k: {} for k in keys}
    sample, warmed = {}, set()

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
            lab, mname = k
            bs, p1, p2, mval = meta[k]
            p = dict(minDisparity=0, numDisparities=ND, blockSize=bs, P1=p1, P2=p2,
                     disp12MaxDiff=D12, preFilterCap=63, uniquenessRatio=UR,
                     speckleWindowSize=SPECKLE[0], speckleRange=SPECKLE[1], mode=mval)
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
            a = agg[k]

            ok = disp >= 0
            err = np.abs(disp - gt)

            e = err[m]
            a["n"] += int(m.sum()); a["bad"] += int(is_bad(e, gt[m]).sum())
            a["err_sum"] += float(e.sum())
            per_scene[k][scene] = 100.0 * float(is_bad(e, gt[m]).mean())

            both = m & ok
            a["nv"] += int(both.sum()); a["badv"] += int(is_bad(err[both], gt[both]).sum())

            a["nc"] += int(m_crop.sum())
            a["badc"] += int(is_bad(err[m_crop], gt[m_crop]).sum())

            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)

            # 小斑块数：有效掩膜上面积小于 SPECKLE_PX 的连通块个数
            # （本步 speckleWindowSize=100 会删掉所有面积 <100 的块，所以三种模式该值均为 0，仅作参考）
            _, _, stats, _ = cv2.connectedComponentsWithStats(ok.astype(np.uint8), connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            a["comp_small"] += int((areas < SPECKLE_PX).sum())

            a["times"].append(float(np.median(times)))

            if scene == VIS_SCENE and lab == "A4":
                sample[k] = disp
                sample["_gt"] = gt
                sample["_vmax"] = float(gt[m].max())

        if (i + 1) % 25 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    rows = []
    for k in keys:
        lab, mname = k
        bs, p1, p2, _ = meta[k]
        a = agg[k]
        d1s = np.array(list(per_scene[k].values()))
        t = float(np.median(a["times"]))
        rows.append(dict(base=lab, mode=mname, bs=bs, p1=p1, p2=p2, scenes=len(scenes),
                         d1_strict=100.0 * a["bad"] / a["n"],
                         d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
                         d1_common=100.0 * a["badc"] / max(a["nc"], 1),
                         epe=a["err_sum"] / a["n"],
                         valid=100.0 * a["valid"] / a["total"],
                         scene_median=float(np.median(d1s)), scene_p90=float(np.percentile(d1s, 90)),
                         scene_worst=float(d1s.max()),
                         comp_small=int(a["comp_small"]), t_med=t, fps=1.0 / t))

    cols = [("基础配置", "base"), ("mode", "mode"), ("blockSize", "bs"), ("P1", "p1"), ("P2", "p2"),
            ("场景数", "scenes"), ("严格D1", "d1_strict"), ("有效区域D1", "d1_validreg"),
            ("公共区域D1", "d1_common"), ("EPE", "epe"), ("有效像素比例", "valid"),
            ("场景D1中位", "scene_median"), ("场景D1_P90", "scene_p90"),
            ("场景D1最差", "scene_worst"), ("小斑块数(面积<100)", "comp_small"),
            ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow(["%.4f" % r[kk] if isinstance(r[kk], float) else r[kk] for _, kk in cols])

    with open(os.path.join(OUT_DIR, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["%s_%s" % (lab, mn) for (lab, mn) in keys])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[k][s] for k in keys])

    print()
    print("%-5s %-10s %-10s %-10s %-10s %-9s %-9s %-9s %-7s"
          % ("配置", "mode", "公共区域D1", "严格D1", "有效区域D1", "有效像素", "P90",
             "小斑块数", "FPS"))
    for r in rows:
        print("%-5s %-10s %-10.2f %-10.2f %-10.2f %-9.1f %-9.2f %-9d %-7.1f"
              % (r["base"], r["mode"], r["d1_common"], r["d1_strict"], r["d1_validreg"],
                 r["valid"], r["scene_p90"], r["comp_small"], r["fps"]))

    print("\n逐场景严格 D1 胜负（基准 = SGBM_3WAY）")
    for (lab, _b, _p1, _p2) in BASES:
        for (mn, _mv) in MODES:
            if mn == "SGBM_3WAY":
                continue
            w1 = w2 = tie = 0
            for s in scenes:
                x, y = per_scene[(lab, mn)][s], per_scene[(lab, "SGBM_3WAY")][s]
                if x < y:
                    w1 += 1
                elif x > y:
                    w2 += 1
                else:
                    tie += 1
            print("  %s %-10s vs SGBM_3WAY : 胜 %3d / 负 %3d / 平 %2d" % (lab, mn, w1, w2, tie))

    # ---------------- 画图 ----------------
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    panels = [("d1_common", "公共区域 D1 (%)"), ("valid", "有效像素比例 (%)"),
              ("fps", "FPS")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colors = {"A4": "#1f77b4", "B4": "#ff7f0e"}
    mnames = [m[0] for m in MODES]
    for ax, (key, title) in zip(axes, panels):
        for (lab, _b, _p1, _p2) in BASES:
            ys = [next(r for r in rows if r["base"] == lab and r["mode"] == mn)[key] for mn in mnames]
            ax.plot(mnames, ys, "o-", color=colors[lab], label=lab)
            offset = (-12, 7) if lab == "A4" else (12, 7)
            for x, y in zip(mnames, ys):
                ax.annotate("%.2f" % y, (x, y), textcoords="offset points",
                            xytext=offset, ha="center", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("第六步 运行模式对比（选参组 %d 场景，其余参数固定）" % len(scenes), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "mode_curves.png"), dpi=130)
    plt.close(fig)

    vmax = sample["_vmax"]
    order = [m[0] for m in MODES]
    fig, axes = plt.subplots(len(order) + 1, 1, figsize=(13, 2.9 * (len(order) + 1)))
    axes[0].imshow(colorize(sample["_gt"], vmax))
    axes[0].set_title("视差真值（disp_noc_0，作参照）", fontsize=11)
    axes[0].axis("off")
    for ax, mn in zip(axes[1:], order):
        r = next(x for x in rows if x["base"] == "A4" and x["mode"] == mn)
        ax.imshow(colorize(sample[("A4", mn)], vmax))
        ax.set_title("%s：mode=%s  │  有效 %.1f%%  公共区域D1 %.2f%%  严格D1 %.2f%%  %.1f FPS"
                     % ("A4", mn, r["valid"], r["d1_common"], r["d1_strict"], r["fps"]), fontsize=11)
        ax.axis("off")
    fig.suptitle("第六步 三种运行模式的视差图对比（场景 %s，A4，共用色标 0~%.0f px）"
                 % (VIS_SCENE, vmax), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "sample_maps_A4.png"), dpi=120)
    plt.close(fig)

    print()
    for name in ("metrics.csv", "per_scene_d1.csv", "mode_curves.png", "sample_maps_A4.png"):
        print("输出 %s" % os.path.join(OUT_DIR, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
