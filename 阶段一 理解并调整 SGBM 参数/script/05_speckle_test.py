# -*- coding: utf-8 -*-
"""
阶段一 · 05 第五步：斑点过滤及删除像素分析
================================================================================

【实验目的】
    在第四步保留的参数基础上比较5组斑点过滤配置，并分析过滤前后发生变化的像素。
    OpenCV的斑点判断同时涉及连通关系和视差差异，因此本实验直接比较关闭过滤与开启
    过滤后的视差结果，不使用普通二值连通块数量代替SGBM实际过滤效果。

【本脚本要回答的问题】
    斑点过滤删掉的像素里，**多少本来就算错（删得对）、多少本来是对的（误删）**。

【基准与判定的定义】按第五步实验方案
    · 基准图 = 关闭过滤 (speckleWindowSize=0) 的视差图
    · 被删除像素 = 基准图有视差、过滤后变成无效的像素
    · 用真值判断这些像素（用**基准图**的视差与真值比较，容差用 D1 判据）：
        原来就是错误视差  → 正确删除
        原来是正确视差    → 误删
    · 另外统计"没被删除但仍然错误" → 漏掉的错误

【真值覆盖率的限制（必须如实分层报告）】
    真值只覆盖约 19~22% 的像素。被删除的像素里大部分没有真值，**无法判定对错**。
    因此删除相关指标分两层给出：
        删除准确率(全体口径) = 正确删除 ÷ 全部被删除像素      ← 方案里的定义，会被真值覆盖率压低
        删除准确率(有真值口径) = 正确删除 ÷ 有真值的被删除像素  ← 可解释的那个
    两个都报，并由 删除总数 / 其中有真值数 两层计数支撑，避免误读。

【配置】第四步保留的 A3 / B3，斑点组合 5 种 → 10 组，150 个选参场景

【输出】
    results/05_speckle_test/metrics.csv          10 组完整指标（含删除分析）
    results/05_speckle_test/per_scene_d1.csv     150 场景 × 10 组
    results/05_speckle_test/deletion_analysis.png 删除分析四联图
    results/05_speckle_test/sample_maps_A3.png / sample_maps_B3.png  代表性视差图
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
BASES = [("A3", 7, 784, 3136), ("B3", 9, 1296, 5184)]
SPECKLE = [(0, 1), (50, 1), (50, 2), (100, 1), (100, 2)]
BASELINE = (0, 1)                 # 关闭过滤 = 基准
REPEATS = 3
FRAME = "10"
VIS_SCENE = "000000"
OUT_DIR = os.path.join(STAGE_DIR, "results", "05_speckle_test")
# ======================================================================


def imread_any(path, flags):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def is_bad(err, gt):
    """KITTI 误匹配判据：绝对误差 > 3 px 且 相对误差 > 5%。"""
    return (err > 3) & (err / gt > 0.05)


def params_of(bs, p1, p2, sw, sr):
    return dict(minDisparity=0, numDisparities=ND, blockSize=bs, P1=p1, P2=p2,
                disp12MaxDiff=D12, preFilterCap=63, uniquenessRatio=UR,
                speckleWindowSize=sw, speckleRange=sr,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def colorize(disp, vmax):
    norm = np.clip(disp / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb[disp < 0] = (0, 0, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def main():
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(GROUP_DIR, "image_2"))
                    if n.endswith(".png"))
    keys = [(lab, sw, sr) for (lab, _b, _p1, _p2) in BASES for (sw, sr) in SPECKLE]
    meta = {(lab, sw, sr): (bs, p1, p2)
            for (lab, bs, p1, p2) in BASES for (sw, sr) in SPECKLE}

    os.makedirs(OUT_DIR, exist_ok=True)
    print("基础配置: " + " | ".join("%s(bs=%d,P1=%d,P2=%d)" % b for b in BASES))
    print("固定: nd=%d uniquenessRatio=%d disp12MaxDiff=%d mode=SGBM_3WAY" % (ND, UR, D12))
    print("斑点组合: %s（(0,1) 为基准）→ %d 组 | 场景 %d 个"
          % (SPECKLE, len(keys), len(scenes)))
    print()

    # 累积器
    agg = {k: dict(n=0, bad=0, err_sum=0.0, nv=0, badv=0, errv_sum=0.0,
                   nc=0, badc=0,
                   valid=0, total=0, times=[],
                   base_valid=0,                 # 基准图有效像素总数（用于算删除比例）
                   deleted=0, deleted_gt=0, deleted_wrong=0, deleted_correct=0,
                   added=0, changed=0,
                   remain_bad=0) for k in keys}
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

        base_map = {}          # 每个基础配置的基准视差图（本场景）
        for k in keys:
            lab, sw, sr = k
            bs, p1, p2 = meta[k]
            sgbm = cv2.StereoSGBM_create(**params_of(bs, p1, p2, sw, sr))
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

            # ---- 常规指标 ----
            e = err[m]
            a["n"] += int(m.sum()); a["bad"] += int(is_bad(e, gt[m]).sum())
            a["err_sum"] += float(e.sum())
            per_scene[k][scene] = 100.0 * float(is_bad(e, gt[m]).mean())

            both = m & ok
            a["nv"] += int(both.sum()); a["badv"] += int(is_bad(err[both], gt[both]).sum())
            a["errv_sum"] += float(err[both].sum())

            a["nc"] += int(m_crop.sum())
            a["badc"] += int(is_bad(err[m_crop], gt[m_crop]).sum())

            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)
            a["times"].append(float(np.median(times)))

            # 漏掉的错误：过滤之后仍然是错误的像素（只统计算法仍有效的）
            a["remain_bad"] += int(is_bad(err[both], gt[both]).sum())

            # ---- 与基准图比较 ----
            if (sw, sr) == BASELINE:
                base_map[lab] = disp
                a["base_valid"] += int(ok.sum())
            else:
                db = base_map[lab]
                vb = db >= 0
                a["base_valid"] += int(vb.sum())
                deleted = vb & (~ok)
                added = (~vb) & ok
                changed = vb & ok & (np.abs(db - disp) > 0)
                a["deleted"] += int(deleted.sum())
                a["added"] += int(added.sum())
                a["changed"] += int(changed.sum())
                # 用基准图的视差与真值比较，判定被删像素原本对错
                dg = deleted & m
                a["deleted_gt"] += int(dg.sum())
                if dg.any():
                    bad_base = is_bad(np.abs(db[dg] - gt[dg]), gt[dg])
                    a["deleted_wrong"] += int(bad_base.sum())
                    a["deleted_correct"] += int((~bad_base).sum())

            if scene == VIS_SCENE:
                sample[k] = disp
                sample["_gt"] = gt
                sample["_vmax"] = float(gt[m].max())

        if (i + 1) % 25 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    # ---------------- 汇总 ----------------
    rows = []
    for k in keys:
        lab, sw, sr = k
        bs, p1, p2 = meta[k]
        a = agg[k]
        d1s = np.array(list(per_scene[k].values()))
        t = float(np.median(a["times"]))
        is_base = (sw, sr) == BASELINE
        rows.append(dict(
            base=lab, sw=sw, sr=sr, bs=bs, p1=p1, p2=p2, scenes=len(scenes),
            d1_common=100.0 * a["badc"] / max(a["nc"], 1),
            d1_strict=100.0 * a["bad"] / a["n"],
            d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
            epe=float(a["err_sum"] / a["n"]),
            epe_valid=float(a["errv_sum"] / max(a["nv"], 1)),
            valid=100.0 * a["valid"] / a["total"],
            scene_median=float(np.median(d1s)), scene_p90=float(np.percentile(d1s, 90)),
            scene_worst=float(d1s.max()),
            # 删除分析
            base_valid=int(a["base_valid"]),
            deleted=int(a["deleted"]),
            deleted_ratio=100.0 * a["deleted"] / max(a["base_valid"], 1),
            deleted_gt=int(a["deleted_gt"]),
            deleted_gt_ratio=100.0 * a["deleted_gt"] / max(a["deleted"], 1),
            deleted_wrong=int(a["deleted_wrong"]),
            deleted_correct=int(a["deleted_correct"]),
            acc_all=100.0 * a["deleted_wrong"] / max(a["deleted"], 1),
            acc_gt=100.0 * a["deleted_wrong"] / max(a["deleted_gt"], 1),
            added=int(a["added"]), changed=int(a["changed"]),
            remain_bad=int(a["remain_bad"]),
            t_med=t, fps=1.0 / t,
        ))

    cols = [("基础配置", "base"), ("speckleWindowSize", "sw"), ("speckleRange", "sr"),
            ("blockSize", "bs"), ("P1", "p1"), ("P2", "p2"), ("场景数", "scenes"),
            ("公共区域D1", "d1_common"), ("严格D1", "d1_strict"), ("有效区域D1", "d1_validreg"),
            ("EPE", "epe"), ("EPE_valid", "epe_valid"), ("有效像素比例", "valid"),
            ("场景D1中位", "scene_median"), ("场景D1_P90", "scene_p90"),
            ("场景D1最差", "scene_worst"),
            ("基准有效像素数", "base_valid"), ("删除像素数", "deleted"),
            ("删除像素比例", "deleted_ratio"), ("删除中有真值数", "deleted_gt"),
            ("删除中有真值比例", "deleted_gt_ratio"),
            ("正确删除数", "deleted_wrong"), ("误删数", "deleted_correct"),
            ("删除准确率_全体", "acc_all"), ("删除准确率_有真值", "acc_gt"),
            ("新增有效数", "added"), ("数值被改动数", "changed"),
            ("漏掉的错误数", "remain_bad"),
            ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow(["%.4f" % r[kk] if isinstance(r[kk], float) else r[kk] for _, kk in cols])

    with open(os.path.join(OUT_DIR, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["%s_sw%d_sr%d" % (lab, sw, sr) for (lab, sw, sr) in keys])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[k][s] for k in keys])

    # ---------------- 控制台 ----------------
    print()
    print("%-4s %-5s %-5s %-10s %-10s %-9s %-9s %-10s %-10s %-9s %-9s"
          % ("配置", "sw", "sr", "公共区域D1", "严格D1", "有效区域D1", "有效像素",
             "删除像素", "删除比例", "正确删除", "误删"))
    for r in rows:
        print("%-4s %-5d %-5d %-10.4f %-10.4f %-9.4f %-9.2f %-10d %-10.4f %-9d %-9d"
              % (r["base"], r["sw"], r["sr"], r["d1_common"], r["d1_strict"], r["d1_validreg"],
                 r["valid"], r["deleted"], r["deleted_ratio"], r["deleted_wrong"],
                 r["deleted_correct"]))

    print()
    print("%-4s %-5s %-5s %-14s %-16s %-16s %-16s %-12s"
          % ("配置", "sw", "sr", "删除中有真值", "删除准确率_全体", "删除准确率_有真值",
             "漏掉的错误", "新增/改动"))
    for r in rows:
        # 关闭过滤（基准行）没有删除任何像素，"删除准确率"是 0/0 未定义，
        # 统一用 — 表示，避免被误读成"准确率 0%"
        is_base = (r["sw"], r["sr"]) == BASELINE
        print("%-4s %-5d %-5d %-14d %-16s %-16s %-16d %-12s"
              % (r["base"], r["sw"], r["sr"], r["deleted_gt"],
                 "—" if is_base else "%.2f" % r["acc_all"],
                 "—" if is_base else "%.2f" % r["acc_gt"],
                 r["remain_bad"], "%d/%d" % (r["added"], r["changed"])))

    # ---------------- 画图 ----------------
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["%d/%d" % (sw, sr) for (sw, sr) in SPECKLE]
    colors = {"A3": "#1f77b4", "B3": "#ff7f0e"}
    panels = [("acc_gt", "删除准确率（有真值口径，%）", "越高说明删得越准"),
              ("deleted_ratio", "删除像素比例（% of 基准有效像素）", "越高说明删得越多"),
              ("d1_common", "公共区域 D1（%）", "整体准确程度"),
              ("valid", "有效像素比例（%）", "还剩多少视差")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (key, title, sub) in zip(axes, panels):
        for (lab, _b, _p1, _p2) in BASES:
            ys = [next(r for r in rows if r["base"] == lab and r["sw"] == sw and r["sr"] == sr)[key]
                  for (sw, sr) in SPECKLE]
            ax.plot(labels, ys, "o-", color=colors[lab], label=lab)
            for x, y in zip(labels, ys):
                ax.annotate("%.2f" % y, (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title("%s\n%s" % (title, sub), fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("speckleWindowSize / speckleRange")
    fig.suptitle("第五步 斑点过滤的删除分析（选参组 %d 场景，ur=0, d12=128）" % len(scenes),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT_DIR, "deletion_analysis.png"), dpi=130)
    plt.close(fig)

    # 代表性视差图
    vmax = sample["_vmax"]
    for (lab, _b, _p1, _p2) in BASES:
        fig, axes = plt.subplots(len(SPECKLE) + 1, 1, figsize=(13, 2.7 * (len(SPECKLE) + 1)))
        axes[0].imshow(colorize(sample["_gt"], vmax))
        axes[0].set_title("视差真值（disp_noc_0，作参照）", fontsize=11)
        axes[0].axis("off")
        for ax, (sw, sr) in zip(axes[1:], SPECKLE):
            r = next(x for x in rows if x["base"] == lab and x["sw"] == sw and x["sr"] == sr)
            tag = "基准（关闭过滤）" if (sw, sr) == BASELINE else "删除 %.2f%%" % r["deleted_ratio"]
            ax.imshow(colorize(sample[(lab, sw, sr)], vmax))
            ax.set_title("%s：speckleWindowSize=%d, speckleRange=%d  │  %s  │  有效 %.2f%%  "
                         "公共区域D1 %.4f%%" % (lab, sw, sr, tag, r["valid"], r["d1_common"]),
                         fontsize=11)
            ax.axis("off")
        fig.suptitle("第五步 斑点过滤对视差图的影响（场景 %s，%s，共用色标 0~%.0f px）"
                     % (VIS_SCENE, lab, vmax), fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUT_DIR, "sample_maps_%s.png" % lab), dpi=120)
        plt.close(fig)

    print()
    for name in ("metrics.csv", "per_scene_d1.csv", "deletion_analysis.png",
                 "sample_maps_A3.png", "sample_maps_B3.png"):
        print("输出 %s" % os.path.join(OUT_DIR, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
