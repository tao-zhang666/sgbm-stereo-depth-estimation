# -*- coding: utf-8 -*-
"""
阶段一 · 02 第二步：联合测试搜索与匹配参数
================================================================================

【这一步在流程里的位置】
    《SGBM最终参数选择方法》§6 第二步：把第一步保留下来的 numDisparities 和 blockSize
    **组合起来**测试，检查两者共同变化时的实际效果，保留综合表现较好的 2~3 组完整配置。
    最终判断针对的是**完整组合**，不是两个参数各自的单项排名。

【候选值来自第一步】
    numDisparities = 64、128、192      （256 被初筛排除）
    blockSize      = 5、7、9           （3 被初筛排除）
    → 3 × 3 = 9 组核心配置

    改 blockSize 时按公式联动 P1/P2（§6 第二步）：P1 = 8*bs*bs，P2 = 32*bs*bs。

【为什么一个脚本里同时算 §5 的两项检查】
    不同 numDisparities 之间存在一个"不公平"：搜索范围越大，左侧固有黑带越宽
    （宽度精确等于 numDisparities）。按"无效算错"的严格口径，大范围会白白多吃亏。
    所以 §5 要求同时报告两种口径，本脚本都算：

      1. 严格 D1        ：全图，算法无效的像素按错误计入（反映实际画面损失）
      2. 公共区域 D1    ：所有候选统一排除左侧相同宽度后再比（只比匹配质量）
                          默认宽度 = 候选里最大的 numDisparities，这样每组都在
                          完全相同的像素集合上被评估；可用 COMMON_CROP 手动指定

    另外统计 §5 要求的"超出搜索范围的真值像素占比"：
      3. 真值超出比例   ：选参组里真值视差大于本组上限（nd-1）的像素占比。
                          它用来判断"某组指标好看，会不会只是把大视差的难点像素
                          直接判成了无效"。
    这三项都只在**选参组**上统计，验收组不参与。

【输出】
    results/02_joint_test/metrics.csv           9 组配置的汇总指标
    results/02_joint_test/per_scene_d1.csv      每个场景每组配置的 D1
    results/02_joint_test/interaction_heatmap.png  参数交互热力图
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
GROUP_DIR = os.path.join(PROJ_ROOT, "data_split", "tune")        # 只用选参组

os.environ.setdefault("MPLCONFIGDIR", os.path.join(STAGE_DIR, ".mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402

# ============================== 实验配置 ==============================
ND_CANDIDATES = [64, 128, 192]      # 第一步保留的 numDisparities
BS_CANDIDATES = [5, 7, 9]           # 第一步保留的 blockSize
COMMON_CROP = None                  # 公共区域排除的左侧宽度；None = 自动取 max(ND_CANDIDATES)
FRAME = "10"
REPEATS = 2                         # 每组配置每个场景计时重复次数（取中位）

BASE = dict(
    minDisparity=0,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
OUT_SUBDIR = "02_joint_test"
# =====================================================================


def imread_any(path, flags):
    """读图，支持含中文路径（cv2.imread 在 Windows 上打不开非 ASCII 路径）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def make_params(nd, bs):
    """按 §6 第二步：改 blockSize 时同步重算 P1/P2。"""
    p = dict(BASE)
    p["numDisparities"] = nd
    p["blockSize"] = bs
    p["P1"] = 8 * 1 * bs * bs
    p["P2"] = 32 * 1 * bs * bs
    return p


def is_bad(err, gt):
    """KITTI 误匹配判据：绝对误差>3px 且 相对误差>5%。"""
    return (err > 3) & (err / gt > 0.05)


def main():
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(GROUP_DIR, "image_2"))
                    if n.endswith(".png"))
    combos = [(nd, bs) for nd in ND_CANDIDATES for bs in BS_CANDIDATES]
    crop = COMMON_CROP if COMMON_CROP is not None else max(ND_CANDIDATES)

    out_dir = os.path.join(STAGE_DIR, "results", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("候选 numDisparities: %s   blockSize: %s" % (ND_CANDIDATES, BS_CANDIDATES))
    print("组合数: %d | 场景数: %d（选参组）" % (len(combos), len(scenes)))
    print("公共区域口径: 排除左侧 %d 列（所有组合统一）" % crop)
    print("固定参数: " + " ".join("%s=%s" % kv for kv in BASE.items()))
    print()

    agg = {c: dict(n=0, bad=0, err_sum=0.0,           # 严格（全图）
                   nv=0, badv=0, errv_sum=0.0,        # 有效区域（全图）
                   nc=0, badc=0, errc_sum=0.0,        # 公共区域（严格）
                   valid=0, total=0,
                   crop_valid=0, crop_total=0,
                   over=0, over_n=0,                  # 真值超出本组范围
                   times=[]) for c in combos}
    per_scene = {c: {} for c in combos}

    for i, scene in enumerate(scenes):
        gl = imread_any(os.path.join(GROUP_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(GROUP_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(GROUP_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        m = gt > 0
        m_crop = m.copy()
        m_crop[:, :crop] = False                      # 公共区域：统一挖掉左侧

        for (nd, bs) in combos:
            sgbm = cv2.StereoSGBM_create(**make_params(nd, bs))
            disp, times = None, []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                d = sgbm.compute(gl, gr).astype(np.float32) / 16.0
                times.append(time.perf_counter() - t0)
                if disp is None:
                    disp = d
            ok = disp >= 0
            err = np.abs(disp - gt)
            a = agg[(nd, bs)]

            # 1) 严格（全图，无效算错）
            e = err[m]
            a["n"] += int(m.sum()); a["bad"] += int(is_bad(e, gt[m]).sum())
            a["err_sum"] += float(e.sum())
            per_scene[(nd, bs)][scene] = 100.0 * float(is_bad(e, gt[m]).mean())

            # 2) 有效区域（全图，只在算法也有效处）
            both = m & ok
            ev = err[both]
            a["nv"] += int(both.sum()); a["badv"] += int(is_bad(ev, gt[both]).sum())
            a["errv_sum"] += float(ev.sum())

            # 3) 公共区域（统一裁剪后，无效仍算错）
            ec = err[m_crop]
            a["nc"] += int(m_crop.sum()); a["badc"] += int(is_bad(ec, gt[m_crop]).sum())
            a["errc_sum"] += float(ec.sum())

            # 4) 完整性
            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)
            a["crop_valid"] += int(ok[:, crop:].sum()); a["crop_total"] += int(ok[:, crop:].size)

            # 5) 真值超出本组搜索范围的像素
            a["over"] += int((gt[m] > (nd - 1)).sum()); a["over_n"] += int(m.sum())

            a["times"].append(float(np.median(times)))

        if (i + 1) % 30 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    # ---------------- 汇总 ----------------
    rows = []
    for (nd, bs) in combos:
        a = agg[(nd, bs)]
        d1s = np.array(list(per_scene[(nd, bs)].values()))
        t = float(np.median(a["times"]))
        rows.append(dict(
            nd=nd, bs=bs, p1=8 * bs * bs, p2=32 * bs * bs, scenes=len(scenes),
            d1_strict=100.0 * a["bad"] / a["n"],
            d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
            d1_common=100.0 * a["badc"] / max(a["nc"], 1),
            epe_strict=a["err_sum"] / a["n"],
            epe_common=a["errc_sum"] / max(a["nc"], 1),
            valid=100.0 * a["valid"] / a["total"],
            valid_common=100.0 * a["crop_valid"] / a["crop_total"],
            over_range=100.0 * a["over"] / a["over_n"],
            scene_mean=float(d1s.mean()), scene_median=float(np.median(d1s)),
            scene_p90=float(np.percentile(d1s, 90)), scene_worst=float(d1s.max()),
            t_med=t, fps=1.0 / t,
        ))

    cols = [("numDisparities", "nd"), ("blockSize", "bs"), ("P1", "p1"), ("P2", "p2"),
            ("场景数", "scenes"),
            ("严格D1", "d1_strict"), ("有效区域D1", "d1_validreg"),
            ("公共区域D1", "d1_common"),
            ("EPE_严格", "epe_strict"), ("EPE_公共区域", "epe_common"),
            ("有效像素比例", "valid"), ("公共区域有效比例", "valid_common"),
            ("真值超出比例", "over_range"),
            ("场景D1均值", "scene_mean"), ("场景D1中位", "scene_median"),
            ("场景D1_P90", "scene_p90"), ("场景D1最差", "scene_worst"),
            ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow(["%.4f" % r[k] if isinstance(r[k], float) else r[k] for _, k in cols])

    with open(os.path.join(out_dir, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["nd%d_bs%d" % (nd, bs) for (nd, bs) in combos])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[(nd, bs)][s] for (nd, bs) in combos])

    # ---------------- 控制台：矩阵形式看交互 ----------------
    def matrix(key, fmt="%7.2f", title=""):
        print("\n%s" % title)
        print("%-14s" % "numD \\ bs" + "".join("%9d" % bs for bs in BS_CANDIDATES))
        for nd in ND_CANDIDATES:
            line = "%-14d" % nd
            for bs in BS_CANDIDATES:
                r = next(x for x in rows if x["nd"] == nd and x["bs"] == bs)
                line += "  " + fmt % r[key]
            print(line)

    print()
    matrix("d1_strict", "%7.2f", "严格 D1（全图，无效算错）")
    matrix("d1_common", "%7.2f", "公共区域 D1（统一排除左侧 %d 列）" % crop)
    matrix("valid", "%7.1f", "有效像素比例 (%)")
    matrix("over_range", "%7.3f", "真值超出本组搜索范围的比例 (%)")
    matrix("fps", "%7.1f", "FPS")

    # ---------------- 画图：交互热力图 ----------------
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    panels = [("d1_strict", "严格 D1（全图，无效算错）", "%.2f", "YlOrRd"),
              ("d1_common", "公共区域 D1（统一裁剪）", "%.2f", "YlOrRd"),
              ("valid", "有效像素比例 (%)", "%.1f", "YlGn"),
              ("over_range", "真值超出搜索范围 (%)", "%.3f", "PuBu")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, (key, title, fmt, cmap) in zip(axes, panels):
        data = np.array([[next(x for x in rows if x["nd"] == nd and x["bs"] == bs)[key]
                          for bs in BS_CANDIDATES] for nd in ND_CANDIDATES])
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(BS_CANDIDATES)), BS_CANDIDATES)
        ax.set_yticks(range(len(ND_CANDIDATES)), ND_CANDIDATES)
        ax.set_xlabel("blockSize")
        ax.set_ylabel("numDisparities")
        ax.set_title(title, fontsize=11)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, fmt % data[r, c], ha="center", va="center", fontsize=10,
                        color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("第二步 联合测试：9 组核心配置（选参组 %d 个场景）" % len(scenes), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = os.path.join(out_dir, "interaction_heatmap.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print()
    print("输出 %s" % os.path.join(out_dir, "metrics.csv"))
    print("输出 %s" % os.path.join(out_dir, "per_scene_d1.csv"))
    print("输出 %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
