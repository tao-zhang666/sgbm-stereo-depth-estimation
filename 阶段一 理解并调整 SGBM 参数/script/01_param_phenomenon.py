# -*- coding: utf-8 -*-
"""
阶段一 · 01 第一步：单参数现象实验（理解参数怎样影响视差图和指标）
================================================================================

【这一步在流程里的位置】
    《SGBM最终参数选择方法》§6 第一步：
        第一步 = 单参数现象实验 —— 只用来**理解规律、排除明显较差的值**，
                 不直接决定最终组合（最终组合由后面的联合测试决定）。

【本脚本做什么】
    固定其它参数，只改一个参数，在**选参组**（data_split/tune，150 个场景）上跑，
    然后同时给出两样东西：

      1. 视差图对比图：同一个代表场景，在这个参数的各个取值下的视差图长什么样
         （配一行真值作参照）——回答"参数变了，图到底怎么变"
      2. 指标曲线图：150 个场景上的 D1 / 有效像素比例 / 中位耗时随取值的变化
         ——回答"参数变了，好坏了多少"

【只碰选参组】
    读取的永远是 data_split/tune/。验收组（data_split/holdout）本脚本不读，
    那是阶段二才用的。

【blockSize 的特殊处理】
    改 blockSize 时，P1/P2 按灰度图公式同步重算（P1 = 8×bs²，P2 = 32×bs²）。
    这样可以与后续联合实验保持相同口径。若不联动，得到的是"P1/P2 固定下"的结论，
    口径与最终选参规则不一致，因此这里一律联动。

【怎么用】
    改下面的 SCAN 选择要扫描的参数，然后运行。每运行一次产出一套文件：
        python 01_param_phenomenon.py
    输出目录按参数名分类：
        results/01_numdisparities_phenomenon/
        results/01_blocksize_phenomenon/
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

# ---- 路径：由本脚本位置推算 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.path.dirname(STAGE_DIR)
GROUP_DIR = os.path.join(PROJ_ROOT, "data_split", "tune")        # 只用 tune 组（选参组；holdout 留到阶段二）

# matplotlib 的字体缓存默认写到用户目录，可能没有写权限，改到项目内
os.environ.setdefault("MPLCONFIGDIR", os.path.join(STAGE_DIR, ".mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager      # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

# ============================== 实验配置 ==============================
SCAN = "numDisparities"        # 要扫描的参数： "numDisparities" 或 "blockSize"

VALUES = {
    "numDisparities": [64, 128, 192, 256],
    "blockSize": [3, 5, 7, 9],
}
COUPLE_P1P2 = True             # 扫描 blockSize 时按公式联动 P1/P2

FRAME = "10"
REPEATS = 3                    # 每个"参数值×场景"计时重复次数，取中位

# 本轮固定的基准参数。除 SCAN 指定的那个参数外，其余全部固定不动。
BASE = dict(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=200,
    P2=800,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

VIS_SCENE = "000000"           # 视差图对比图用哪个场景（随便挑一个选参组里的场景）
# =====================================================================


def make_params(scan, value):
    """在基准参数上改一个参数，返回完整参数组合。"""
    p = dict(BASE)
    p[scan] = value
    if scan == "blockSize" and COUPLE_P1P2:
        # 灰度图通道数=1，所以就是 8×bs² 和 32×bs²
        p["P1"] = 8 * 1 * value * value
        p["P2"] = 32 * 1 * value * value
    return p


def list_scenes():
    """选参组里有哪些场景：以 image_2 目录下的文件为准。"""
    d = os.path.join(GROUP_DIR, "image_2")
    if not os.path.isdir(d):
        raise SystemExit("找不到选参组目录，请先运行 00_make_split.py 创建数据划分：%s" % d)
    return sorted(n[:6] for n in os.listdir(d) if n.endswith(".png"))


def imread_any(path, flags):
    """
    读取图片，**支持含中文的路径**。

    为什么不能直接用 cv2.imread：
        OpenCV 在 Windows 上打不开非 ASCII 路径，实测直接返回 None（不抛异常），
        项目所在路径或上级目录可能包含中文字符，因此统一采用这种读取方式。
    绕法：先用 numpy 读成字节，再交给 cv2.imdecode 解码。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def read_gray(scene, cam):
    p = os.path.join(GROUP_DIR, cam, "%s_%s.png" % (scene, FRAME))
    img = imread_any(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("读图失败: %s" % p)
    return img


def read_gt(scene, kind):
    p = os.path.join(GROUP_DIR, kind, "%s_%s.png" % (scene, FRAME))
    # 必须 IMREAD_UNCHANGED，否则 16 位真值会被截断成 8 位
    raw = imread_any(p, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit("读真值失败: %s" % p)
    return raw.astype(np.float32) / 256.0


def colorize(disp, vmax):
    """视差 -> 伪彩；无效像素(<0)涂黑；超过 vmax 压到最高色。"""
    norm = np.clip(disp / vmax * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[disp < 0] = (0, 0, 0)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def is_bad(err, gt):
    """KITTI 误匹配判据：绝对误差 > 3 px 且 相对误差 > 5%。"""
    return (err > 3) & (err / gt > 0.05)


def setup_cjk_font():
    """让 matplotlib 能显示中文。"""
    names = [f.name for f in font_manager.fontManager.ttflist]
    for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    # 命令行可以临时指定要扫描的参数，省得改文件：
    #     python 01_param_phenomenon.py blockSize
    global SCAN
    if len(sys.argv) > 1:
        SCAN = sys.argv[1]

    if SCAN not in VALUES:
        raise SystemExit("SCAN 只能是 %s 之一" % list(VALUES))
    values = VALUES[SCAN]
    scenes = list_scenes()
    out_dir = os.path.join(STAGE_DIR, "results", "01_%s_phenomenon" % SCAN.lower())
    os.makedirs(out_dir, exist_ok=True)

    print("扫描参数: %s  取值: %s" % (SCAN, values))
    print("数据: 选参组 %d 个场景（%s）" % (len(scenes), GROUP_DIR))
    print("固定参数: " + " ".join(
        "%s=%s" % (k, v) for k, v in BASE.items() if k != SCAN))
    if SCAN == "blockSize" and COUPLE_P1P2:
        # 提示语里不用上标符号 ²，Windows 控制台默认 GBK 编码打不出来会直接报错
        print("注意: P1/P2 随 blockSize 联动（P1 = 8*bs*bs, P2 = 32*bs*bs）")
    print()

    # 累积器。pooled 指标要累加像素个数，不能先算每场景比例再平均
    agg = {v: dict(err_sum=0.0, n=0, bad=0,                # 严格口径（无效算错）
                   errv_sum=0.0, nv=0, badv=0,             # 只有效区域
                   bad_occ=0, n_occ=0,
                   valid=0, total=0, times=[]) for v in values}
    per_scene = {v: {} for v in values}

    # 代表场景的图与指标（画对比图用）
    vis = {}

    for idx, scene in enumerate(scenes):
        gl = read_gray(scene, "image_2")
        gr = read_gray(scene, "image_3")
        gt_noc = read_gt(scene, "disp_noc_0")
        gt_occ = read_gt(scene, "disp_occ_0")
        m_noc, m_occ = gt_noc > 0, gt_occ > 0

        for v in values:
            params = make_params(SCAN, v)
            sgbm = cv2.StereoSGBM_create(**params)
            disp, times = None, []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                d = sgbm.compute(gl, gr).astype(np.float32) / 16.0
                times.append(time.perf_counter() - t0)
                if disp is None:
                    disp = d

            ok = disp >= 0
            a = agg[v]

            err = np.abs(disp - gt_noc)
            e = err[m_noc]
            a["err_sum"] += float(e.sum()); a["n"] += int(m_noc.sum())
            a["bad"] += int(is_bad(e, gt_noc[m_noc]).sum())

            both = m_noc & ok
            ev = err[both]
            a["errv_sum"] += float(ev.sum()); a["nv"] += int(both.sum())
            a["badv"] += int(is_bad(ev, gt_noc[both]).sum())

            per_scene[v][scene] = 100.0 * float(is_bad(e, gt_noc[m_noc]).mean())

            e_occ = np.abs(disp - gt_occ)[m_occ]
            a["bad_occ"] += int(is_bad(e_occ, gt_occ[m_occ]).sum())
            a["n_occ"] += int(m_occ.sum())

            a["valid"] += int(ok.sum()); a["total"] += int(ok.size)
            a["times"].append(float(np.median(times)))

            if scene == VIS_SCENE:
                vis.setdefault(v, {})
                vis[v] = dict(disp=disp, median=float(np.median(e)),
                              d1=per_scene[v][scene], valid=100.0 * ok.mean(),
                              t=float(np.median(times)))
                vis["gt_max"] = float(gt_noc[m_noc].max())
                vis["gt"] = gt_noc
                vis["gt_min"] = float(gt_noc[m_noc].min())

        if (idx + 1) % 30 == 0:
            print("  ... 已处理 %d/%d 个场景" % (idx + 1, len(scenes)))

    # ---------------- 汇总 ----------------
    rows = []
    for v in values:
        a = agg[v]
        d1s = np.array(list(per_scene[v].values()))
        rows.append(dict(
            param=SCAN, value=v, scenes=len(scenes),
            d1_strict=100.0 * a["bad"] / a["n"],
            d1_validreg=100.0 * a["badv"] / max(a["nv"], 1),
            d1_occ=100.0 * a["bad_occ"] / a["n_occ"],
            scene_mean=float(d1s.mean()), scene_median=float(np.median(d1s)),
            scene_p90=float(np.percentile(d1s, 90)), scene_worst=float(d1s.max()),
            epe=float(a["err_sum"] / a["n"]),
            valid=100.0 * a["valid"] / a["total"],
            t_med=float(np.median(a["times"])),
            fps=1.0 / float(np.median(a["times"])),
        ))

    # ---- 指标表 CSV ----
    cols = [("参数", "param"), ("取值", "value"), ("场景数", "scenes"),
            ("合并D1_严格", "d1_strict"), ("合并D1_有效区域", "d1_validreg"),
            ("合并D1_严格_occ", "d1_occ"), ("场景D1均值", "scene_mean"),
            ("场景D1中位", "scene_median"), ("场景D1_P90", "scene_p90"),
            ("场景D1最差", "scene_worst"), ("EPE", "epe"),
            ("有效像素比例", "valid"), ("中位耗时s", "t_med"), ("FPS", "fps")]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow(["%.4f" % r[k] if isinstance(r[k], float) else r[k] for _, k in cols])

    with open(os.path.join(out_dir, "per_scene_d1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号"] + ["%s=%s" % (SCAN, v) for v in values])
        for s in scenes:
            w.writerow([s] + ["%.2f" % per_scene[v][s] for v in values])

    # ---- 控制台表 ----
    print()
    print("%-6s %-12s %-14s %-9s %-9s %-9s %-9s %-8s %-7s"
          % ("取值", "严格D1", "有效区域D1", "D1_occ", "场景均值",
             "场景中位", "场景P90", "有效%", "FPS"))
    for r in rows:
        print("%-6s %-12.2f %-14.2f %-9.2f %-9.2f %-9.2f %-9.2f %-8.1f %-7.1f"
              % (r["value"], r["d1_strict"], r["d1_validreg"], r["d1_occ"],
                 r["scene_mean"], r["scene_median"], r["scene_p90"], r["valid"], r["fps"]))

    # ---------------- 画图 ----------------
    setup_cjk_font()
    xs = [str(v) for v in values]

    # 图 1：代表场景的视差图对比（每个取值一行 + 真值一行）
    vmax = vis["gt_max"]
    order = list(values) + ["GT"]
    fig, axes = plt.subplots(len(order), 1, figsize=(13, 2.9 * len(order)))
    for ax, key in zip(axes, order):
        if key == "GT":
            ax.imshow(colorize(vis["gt"], vmax))
            ax.set_title("视差真值（disp_noc_0，作参照）　范围 %.1f ~ %.1f px"
                         % (vis["gt_min"], vmax), fontsize=11)
        else:
            d = vis[key]
            ax.imshow(colorize(d["disp"], vmax))
            ax.set_title("%s = %s　│ 有效 %.1f%%　中值误差 %.2f px　D1 %.1f%%　中位耗时 %.3f s"
                         % (SCAN, key, d["valid"], d["median"], d["d1"], d["t"]), fontsize=11)
        ax.axis("off")
    fig.suptitle("单参数现象实验：%s 对视差图的影响（场景 %s，共用色标 0~%.0f px）"
                 % (SCAN, VIS_SCENE, vmax), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(out_dir, "disparity_maps_%s.png" % VIS_SCENE)
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 图 2：指标曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(xs, [r["d1_strict"] for r in rows], "o-", label="严格 D1（无效算错）")
    ax.plot(xs, [r["d1_validreg"] for r in rows], "s--", label="有效区域 D1")
    ax.plot(xs, [r["d1_occ"] for r in rows], "^:", label="含遮挡真值 D1")
    for r in rows:
        ax.annotate("%.1f" % r["d1_strict"], (str(r["value"]), r["d1_strict"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel(SCAN); ax.set_ylabel("D1 误匹配率 (%)")
    ax.set_title("精度"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(xs, [r["valid"] for r in rows], "o-", color="#d62728")
    for r in rows:
        ax.annotate("%.1f" % r["valid"], (str(r["value"]), r["valid"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel(SCAN); ax.set_ylabel("有效像素比例 (%)")
    ax.set_title("完整性"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(xs, [r["t_med"] * 1000 for r in rows], color="#7f7f7f")
    for r in rows:
        ax.annotate("%.0f FPS" % r["fps"], (str(r["value"]), r["t_med"] * 1000),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xlabel(SCAN); ax.set_ylabel("中位单帧耗时 (ms)")
    ax.set_title("速度"); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("单参数现象实验：%s 对指标的影响（选参组 %d 个场景）" % (SCAN, len(scenes)),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = os.path.join(out_dir, "metrics_curves.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    print()
    print("输出 %s" % os.path.join(out_dir, "metrics.csv"))
    print("输出 %s" % os.path.join(out_dir, "per_scene_d1.csv"))
    print("输出 %s" % p1)
    print("输出 %s" % p2)


if __name__ == "__main__":
    main()
