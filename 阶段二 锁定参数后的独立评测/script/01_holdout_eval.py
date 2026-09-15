# -*- coding: utf-8 -*-
"""
阶段二 · 01 锁定参数后的独立评测（真正上场）
================================================================================

【本阶段要回答的问题】
    阶段一在 150 个选参场景上选出并锁定了唯一一组参数（configs/sgbm.yaml，配置 B5）。
    本脚本**只使用另外 50 个从未参与调参的验收场景**（data_split/holdout），
    判断这组参数换到没见过的图片上还能不能保持效果。

    独立评测规则：
      · 参数在验收期间保持不变，本脚本不从命令行接收任何参数覆盖；
      · 不把验收场景重新用于调参；验收结果不得反过来修改参数；
      · 如果验收失败，应回到阶段一重新选参，再重新完整验收。

【指标】D1 误匹配率、EPE、中值误差、有效视差比例、单帧中位耗时与 FPS
    口径与阶段一保持一致，便于对照：
      · 公共区域 D1：真值>0 且排除左侧 numDisparities 列；算法没作答按错误计入
      · 全图严格 D1：真值>0 的像素；算法没作答按错误计入
      · 有效区域 D1：真值>0 且算法作答
      · D1-all（补充）：用 disp_occ_0（含遮挡真值）算的全图严格 D1，仅作参考
      · 有效视差比例：算法作答的像素 / 全部像素
      · 真值区域有效率：算法作答的像素 / 真值>0 的像素（衡量"该给答案的地方给了没有"）
      · 耗时：预热一次后再重复 REPEATS 次取中位

【配置读取】
    优先用 pyyaml；如果环境里没装（本项目 .venv 目前就没有），
    退回下面这个只解析本配置文件所用子集（一层映射 + 标量）的极简读取器。

【输出】
    阶段二 锁定参数后的独立评测/results/holdout_eval/
        metrics.csv                 总体指标
        per_scene.csv               50 个场景逐场景指标
        vs_stage1.csv               与阶段一选参结果的对照
        error_visualization.png     代表性场景误差图（最好 / 中位 / 最差）
        scene_d1_sorted.png         50 个场景 D1 排序（看尾部）
        cases/failure_*.png         失败案例（最差的 3 个场景）
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
HOLDOUT_DIR = os.path.join(PROJ_ROOT, "data_split", "holdout")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "sgbm.yaml")
STAGE1_RESULTS = os.path.join(PROJ_ROOT, "阶段一 理解并调整 SGBM 参数", "results",
                              "06_mode_test", "metrics.csv")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(STAGE_DIR, ".mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402

REPEATS = 3
FRAME = "10"
OUT_DIR = os.path.join(STAGE_DIR, "results", "holdout_eval")
CASES_DIR = os.path.join(OUT_DIR, "cases")

MODE_MAP = {"SGBM": cv2.STEREO_SGBM_MODE_SGBM,
            "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            "HH": cv2.STEREO_SGBM_MODE_HH}


# ----------------------------------------------------------------- config
def _mini_yaml(path):
    """只解析本配置文件用到的子集：一层映射 + 标量值，忽略注释与空行。"""
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


def load_config(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f), "pyyaml"
    except ImportError:
        return _mini_yaml(path), "内置极简解析（环境里没有 pyyaml）"


def read_config():
    cfg, how = load_config(CONFIG_PATH)
    p = dict(cfg["sgbm"])
    p["mode"] = MODE_MAP[p["mode"]]
    return p, how


# ----------------------------------------------------------------- helpers
def imread_any(path, flags):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def is_bad(err, gt):
    return (err > 3) & (err / gt > 0.05)


def bad_map(disp, gt):
    """D1 判错：|d-gt|>3 px 且相对误差>5%；**算法没作答的像素也按错误计入**。

    显式写出"没作答即错误"，不再依赖 |(-1)-gt| 这个算式。
    （实测本项目真值像素最小值为 4，因此两种写法结果相同；
      写清楚是为了不留下隐式假设。）
    """
    e = np.abs(disp - gt)
    safe = np.where(gt > 0, gt, 1.0)        # gt=0 的像素不参与判错，避免除零
    return ((e > 3) & (e / safe > 0.05)) | (disp < 0)


def colorize(disp, vmax):
    norm = np.clip(disp / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb[disp < 0] = (0, 0, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def err_map(disp, gt, vmax=10.0):
    """误差可视化。

    KITTI 真值稀疏（约 20% 像素有真值），所以必须区分三种位置：
        黑 = 没有真值        → 无从评价，不能算算法错误
        灰 = 有真值但算法无输出 → 按错误计入（严格口径）
        彩色 = 有真值且有输出 → 按 |d-gt| 着色（0~vmax px）
    没有真值的位置必须单独显示为黑色，不能根据预测值为其计算或显示误差。
    """
    e = np.abs(disp - gt)
    norm = np.clip(e / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    rgb[gt <= 0] = (0, 0, 0)                      # 无真值
    rgb[(gt > 0) & (disp < 0)] = (128, 128, 128)  # 有真值但无输出
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def main():
    params, how = read_config()
    nd = params["numDisparities"]
    scenes = sorted(n[:6] for n in os.listdir(os.path.join(HOLDOUT_DIR, "image_2"))
                    if n.endswith(".png"))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CASES_DIR, exist_ok=True)

    print("=" * 78)
    print("阶段二 · 锁定参数后的独立评测")
    print("=" * 78)
    print("参数来源: %s（读取方式：%s）" % (CONFIG_PATH, how))
    for k in ("minDisparity", "numDisparities", "blockSize", "P1", "P2", "disp12MaxDiff",
              "preFilterCap", "uniquenessRatio", "speckleWindowSize", "speckleRange", "mode"):
        print("    %-18s %s" % (k, params[k]))
    print("验收场景: %s（%d 个，未参与任何调参）" % (HOLDOUT_DIR, len(scenes)))
    print("提示：本脚本不接收参数覆盖，参数在验收期间保持不变")
    print()

    # ---- 逐场景 ----
    per = []
    reps = {}
    errs_gt = []          # 汇总全部真值像素的误差，用于计算真正的"整体中值误差"
    for i, scene in enumerate(scenes):
        gl = imread_any(os.path.join(HOLDOUT_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(HOLDOUT_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(HOLDOUT_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        gto = imread_any(os.path.join(HOLDOUT_DIR, "disp_occ_0", "%s_%s.png" % (scene, FRAME)),
                         cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0

        sgbm = cv2.StereoSGBM_create(**params)
        sgbm.compute(gl, gr)                       # 预热，不计时

        disp, times = None, []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            d = sgbm.compute(gl, gr).astype(np.float32) / 16.0
            times.append(time.perf_counter() - t0)
            if disp is None:
                disp = d
        t_med = float(np.median(times))

        ok = disp >= 0
        err = np.abs(disp - gt)
        m = gt > 0
        m_crop = m.copy()
        m_crop[:, :nd] = False
        mo = gto > 0

        both = m & ok
        bad_noc = bad_map(disp, gt)          # 用非遮挡真值判错
        bad_occ = bad_map(disp, gto)         # 用含遮挡真值判错（必须用同一套真值算误差！）
        r = dict(
            scene=scene,
            n_gt=int(m.sum()),
            n_gt_crop=int(m_crop.sum()),
            n_gt_occ=int(mo.sum()),
            n_valid=int(ok.sum()),
            n_px=int(ok.size),
            n_valid_gt=int(both.sum()),
            n_both=int(both.sum()),
            d1_strict=100.0 * float(bad_noc[m].mean()),
            d1_common=100.0 * float(bad_noc[m_crop].mean()),
            d1_validreg=100.0 * float(is_bad(err[both], gt[both]).mean()) if both.any() else float("nan"),
            d1_occ=100.0 * float(bad_occ[mo].mean()),
            epe=float(err[m].mean()),
            epe_valid=float(err[both].mean()) if both.any() else float("nan"),
            med_err=float(np.median(err[m])),
            med_err_valid=float(np.median(err[both])) if both.any() else float("nan"),
            valid_ratio=100.0 * float(ok.mean()),
            valid_ratio_gt=100.0 * float(ok[m].mean()),
            t_med=t_med, fps=1.0 / t_med,
        )
        per.append(r)
        errs_gt.append(err[m].astype(np.float32))
        if scene in (scenes[0],):
            reps["disp"] = disp
            reps["gt"] = gt
            reps["left"] = gl
            reps["vmax"] = float(gt[m].max())

        if (i + 1) % 10 == 0:
            print("  ... 已处理 %d/%d 个场景" % (i + 1, len(scenes)))

    # ---- 汇总 ----
    def agg(key, w):
        """按像素数加权汇总（pooled）：Σ(逐场景值 × 像素数) / Σ像素数。"""
        return float(np.sum([x[key] * x[w] for x in per]) / np.sum([x[w] for x in per]))

    total_gt = sum(x["n_gt"] for x in per)
    total_gt_crop = sum(x["n_gt_crop"] for x in per)
    total_px = sum(x["n_px"] for x in per)
    total_valid = sum(x["n_valid"] for x in per)
    total_valid_gt = sum(x["n_valid_gt"] for x in per)
    all_err = np.concatenate(errs_gt)          # 全部真值像素的误差（无输出按 gt+1 计入）
    summary = dict(
        scenes=len(per),
        d1_strict=agg("d1_strict", "n_gt"),
        d1_common=agg("d1_common", "n_gt_crop"),
        # 有效区域口径：分母是"既有真值、又有输出"的像素，因此必须用 n_both 加权
        d1_validreg=agg("d1_validreg", "n_both"),
        d1_occ=agg("d1_occ", "n_gt_occ"),
        epe=agg("epe", "n_gt"),
        epe_pooled=float(all_err.mean()),      # 自检：应与 epe 相同
        epe_valid=agg("epe_valid", "n_both"),
        # 中值误差主口径 = 全部真值像素汇总后的中位数（不是"各场景中值的中位数"）
        med_err=float(np.median(all_err)),
        med_err_scene=float(np.median([x["med_err"] for x in per])),
        med_err_valid=float(np.median([x["med_err_valid"] for x in per])),
        # 比例类指标按像素汇总（与阶段一的口径一致），不是各场景平均
        valid_ratio=100.0 * total_valid / total_px,
        valid_ratio_gt=100.0 * total_valid_gt / total_gt,
        t_med=float(np.median([x["t_med"] for x in per])),
        fps=1.0 / float(np.median([x["t_med"] for x in per])),
        scene_median=float(np.median([x["d1_strict"] for x in per])),
        scene_p90=float(np.percentile([x["d1_strict"] for x in per], 90)),
        scene_worst=float(max(x["d1_strict"] for x in per)),
        gt_pixels=total_gt, gt_pixels_crop=total_gt_crop,
    )

    print()
    print("---- %d 个验收场景总体结果 ----" % len(per))
    print("  公共区域 D1      %8.4f %%" % summary["d1_common"])
    print("  全图严格 D1      %8.4f %%" % summary["d1_strict"])
    print("  有效区域 D1      %8.4f %%" % summary["d1_validreg"])
    print("  D1-all（含遮挡）  %8.4f %%" % summary["d1_occ"])
    print("  EPE              %8.4f px（全像素汇总自检 %.4f）"
          % (summary["epe"], summary["epe_pooled"]))
    print("  EPE（有效区域）    %8.4f px" % summary["epe_valid"])
    print("  中值误差（全部真值像素）    %8.4f px" % summary["med_err"])
    print("  中值误差（各场景中值的中位）%8.4f px" % summary["med_err_scene"])
    print("  有效视差比例      %8.4f %%" % summary["valid_ratio"])
    print("  真值区域有效率    %8.4f %%" % summary["valid_ratio_gt"])
    print("  中位耗时          %8.4f s（%.2f FPS）" % (summary["t_med"], summary["fps"]))
    print("  场景 D1 中位/P90/最差: %.4f / %.4f / %.4f %%"
          % (summary["scene_median"], summary["scene_p90"], summary["scene_worst"]))

    # ---- 与阶段一对照 ----
    vs = []
    if os.path.isfile(STAGE1_RESULTS):
        s1rows = list(csv.DictReader(open(STAGE1_RESULTS, encoding="utf-8-sig")))
        s1 = next(r for r in s1rows if r["mode"] == "SGBM_3WAY" and r["基础配置"] == "B4")
        pairs = [("公共区域D1", s1["公共区域D1"], summary["d1_common"]),
                 ("全图严格D1", s1["严格D1"], summary["d1_strict"]),
                 ("有效区域D1", s1["有效区域D1"], summary["d1_validreg"]),
                 ("EPE", s1["EPE"], summary["epe"]),
                 ("有效视差比例", s1["有效像素比例"], summary["valid_ratio"]),
                 ("中位耗时s", s1["中位耗时s"], summary["t_med"])]
        print()
        print("---- 与阶段一（150 选参场景）对照 ----")
        print("  %-14s %12s %12s %10s" % ("指标", "阶段一(选参)", "阶段二(验收)", "变化"))
        for name, a, b in pairs:
            a, b = float(a), float(b)
            print("  %-14s %12.4f %12.4f %+10.4f" % (name, a, b, b - a))
            vs.append(dict(指标=name, 阶段一选参=a, 阶段二验收=b, 变化=b - a))

    # ---- 写 CSV ----
    with open(os.path.join(OUT_DIR, "per_scene.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        cols = ["场景编号", "公共区域D1", "全图严格D1", "有效区域D1", "D1_含遮挡", "EPE",
                "EPE_有效区域", "中值误差", "中值误差_有效", "有效视差比例", "真值区域有效率",
                "中位耗时s", "FPS", "真值像素数"]
        w.writerow(cols)
        for x in per:
            w.writerow([x["scene"], "%.4f" % x["d1_common"], "%.4f" % x["d1_strict"],
                        "%.4f" % x["d1_validreg"], "%.4f" % x["d1_occ"], "%.4f" % x["epe"],
                        "%.4f" % x["epe_valid"], "%.4f" % x["med_err"],
                        "%.4f" % x["med_err_valid"], "%.4f" % x["valid_ratio"],
                        "%.4f" % x["valid_ratio_gt"], "%.4f" % x["t_med"], "%.2f" % x["fps"],
                        x["n_gt"]])

    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["配置", "验收场景数", "公共区域D1", "全图严格D1", "有效区域D1", "D1_含遮挡",
                    "EPE", "EPE_有效区域", "中值误差", "中值误差_各场景中值", "有效视差比例",
                    "真值区域有效率", "场景D1中位", "场景D1_P90", "场景D1最差", "中位耗时s", "FPS"])
        w.writerow(["B5（configs/sgbm.yaml）", len(per),
                    "%.4f" % summary["d1_common"], "%.4f" % summary["d1_strict"],
                    "%.4f" % summary["d1_validreg"], "%.4f" % summary["d1_occ"],
                    "%.4f" % summary["epe"], "%.4f" % summary["epe_valid"],
                    "%.4f" % summary["med_err"], "%.4f" % summary["med_err_scene"],
                    "%.4f" % summary["valid_ratio"],
                    "%.4f" % summary["valid_ratio_gt"], "%.4f" % summary["scene_median"],
                    "%.4f" % summary["scene_p90"], "%.4f" % summary["scene_worst"],
                    "%.4f" % summary["t_med"], "%.2f" % summary["fps"]])

    if vs:
        with open(os.path.join(OUT_DIR, "vs_stage1.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["指标", "阶段一_150选参场景", "阶段二_50验收场景", "变化"])
            for d in vs:
                w.writerow([d["指标"], "%.4f" % d["阶段一选参"], "%.4f" % d["阶段二验收"],
                            "%+.4f" % d["变化"]])

    # ---- 画图 ----
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False

    order = sorted(per, key=lambda x: x["d1_strict"])
    picks = [order[0], order[len(order) // 2], order[-1]]
    titles = ["最好（%s）", "中位（%s）", "最差（%s）"]
    vmax = reps["vmax"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    for row, (rec, ttl) in enumerate(zip(picks, titles)):
        scene = rec["scene"]
        gl = imread_any(os.path.join(HOLDOUT_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(HOLDOUT_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(HOLDOUT_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp = cv2.StereoSGBM_create(**params).compute(gl, gr).astype(np.float32) / 16.0
        axes[row, 0].imshow(gl, cmap="gray")
        axes[row, 0].set_title((ttl % scene) + " · 左图", fontsize=10)
        axes[row, 1].imshow(colorize(disp, vmax))
        axes[row, 1].set_title("SGBM 视差（B5 锁定参数）", fontsize=10)
        axes[row, 2].imshow(err_map(disp, gt, 10.0))
        axes[row, 2].set_title("误差 |d-gt|（0~10 px）｜黑=无真值，灰=有真值但无输出",
                               fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("阶段二 独立验收：代表性场景（%d 个验收场景中的最好 / 中位 / 最差）" % len(per), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "error_visualization.png"), dpi=120)
    plt.close(fig)

    # D1 排序条形图
    order_asc = sorted(per, key=lambda x: -x["d1_strict"])
    fig, ax = plt.subplots(figsize=(14, 5))
    xs = ["%s" % x["scene"] for x in order_asc]
    ys = [x["d1_strict"] for x in order_asc]
    colors = ["#d62728" if i < 3 else "#1f77b4" for i in range(len(ys))]
    ax.bar(range(len(ys)), ys, color=colors)
    ax.set_xticks(range(len(xs)), xs, rotation=90, fontsize=7)
    ax.set_ylabel("全图严格 D1 (%)")
    ax.set_title("阶段二 独立验收：%d 个验收场景的全图严格 D1（红 = 最差 3 个，另存为失败案例）" % len(per))
    if vs:
        s1_d1 = float(next(d["阶段一选参"] for d in vs if d["指标"] == "全图严格D1"))
        ax.axhline(s1_d1, color="#2ca02c", ls="--", lw=1.5,
                   label="阶段一选参集（150 场景）严格 D1 = %.2f%%" % s1_d1)
        ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scene_d1_sorted.png"), dpi=130)
    plt.close(fig)

    # 失败案例
    for rank, rec in enumerate(order[-3:][::-1], start=1):
        scene = rec["scene"]
        gl = imread_any(os.path.join(HOLDOUT_DIR, "image_2", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gr = imread_any(os.path.join(HOLDOUT_DIR, "image_3", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_GRAYSCALE)
        gt = imread_any(os.path.join(HOLDOUT_DIR, "disp_noc_0", "%s_%s.png" % (scene, FRAME)),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp = cv2.StereoSGBM_create(**params).compute(gl, gr).astype(np.float32) / 16.0
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
        axes[0].imshow(gl, cmap="gray"); axes[0].set_title("左图", fontsize=11)
        axes[1].imshow(colorize(gt, vmax)); axes[1].set_title("视差真值", fontsize=11)
        axes[2].imshow(colorize(disp, vmax)); axes[2].set_title("SGBM 视差", fontsize=11)
        axes[3].imshow(err_map(disp, gt, 10.0))
        axes[3].set_title("误差 |d-gt|（黑=无真值，灰=有真值但无输出）", fontsize=11)
        for ax in axes:
            ax.axis("off")
        fig.suptitle("失败案例 %d：场景 %s ｜ 全图严格 D1 %.2f%% ｜ 有效视差 %.2f%%"
                     % (rank, scene, rec["d1_strict"], rec["valid_ratio"]), fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(CASES_DIR, "failure_%d_%s.png" % (rank, scene)), dpi=120)
        plt.close(fig)

    print()
    for name in ("metrics.csv", "per_scene.csv", "vs_stage1.csv",
                 "error_visualization.png", "scene_d1_sorted.png"):
        print("输出 %s" % os.path.join(OUT_DIR, name))
    print("输出 %s" % os.path.join(CASES_DIR, "failure_*.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
