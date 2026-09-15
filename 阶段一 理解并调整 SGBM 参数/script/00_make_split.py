# -*- coding: utf-8 -*-
r"""
阶段一 · 00 数据划分：把 KITTI 训练集的 200 个场景固定拆成选参组 150 + 验收组 50
================================================================================

【为什么需要这个脚本】
    阶段一在 150 个选参场景上选参数，阶段二在另外 50 个从未参与选参的场景上做独立评测。
    这个划分必须可复现，否则不同运行得到的场景分组和评测结果将无法直接对照。

【划分规则】
    1. 取训练集全部场景号：000000 ~ 000199（按字符串排序）；
    2. 用 Python 标准库 random.Random(42) 做一次 shuffle；
    3. shuffle 之后前 150 个为选参组（tune），末尾 50 个为验收组（holdout）。

    仓库中的 data_split/scene_split.csv 记录了本项目实际使用的完整分组。
    复现实验时应同时核对生成结果与这份清单。

【它做什么】
    · 从 data_scene_flow/training 复制每个场景的
      image_2 / image_3 / disp_noc_0 / disp_occ_0 中的 `_10.png` 文件；
    · 创建 data_split/tune 和 data_split/holdout；
    · 写出 data_split/scene_split.csv；
    · 检查四个子目录的文件是否齐全、是否混入了其他场景。

【用法】
    python "阶段一 理解并调整 SGBM 参数\script\00_make_split.py"          # 创建或补齐数据划分
    python "阶段一 理解并调整 SGBM 参数\script\00_make_split.py" --check  # 只核对，不复制或修改文件
"""

import argparse
import csv
import os
import random
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.path.dirname(STAGE_DIR)
SRC = os.path.join(PROJ_ROOT, "data_scene_flow", "training")
DST = os.path.join(PROJ_ROOT, "data_split")
SUBDIRS = ("image_2", "image_3", "disp_noc_0", "disp_occ_0")
FRAME = "10"
SEED = 42
N_HOLDOUT = 50


def split_ids():
    """返回固定的 (tune_ids, holdout_ids)。"""
    ids = sorted("%06d" % i for i in range(200))
    random.Random(SEED).shuffle(ids)
    return ids[:-N_HOLDOUT], ids[-N_HOLDOUT:]


def expected_files(ids):
    return {"%s_%s.png" % (sid, FRAME) for sid in ids}


def png_files(path):
    if not os.path.isdir(path):
        return None
    return {name for name in os.listdir(path) if name.lower().endswith(".png")}


def describe_difference(group, subdir, got, want):
    """打印一个子目录相对预期清单的差异，并返回是否一致。"""
    label = "data_split/%s/%s" % (group, subdir)
    if got is None:
        print("  [缺少] %s" % label)
        return False

    missing = sorted(want - got)
    extra = sorted(got - want)
    if not missing and not extra:
        print("  [通过] %-34s %d 个文件" % (label, len(got)))
        return True

    print("  [不一致] %s：现有 %d，缺少 %d，多出 %d"
          % (label, len(got), len(missing), len(extra)))
    if missing:
        print("           缺少示例：%s" % ", ".join(missing[:5]))
    if extra:
        print("           多出示例：%s" % ", ".join(extra[:5]))
    return False


def check_split(tune, holdout):
    """完整检查两个分组的四个子目录。"""
    ok = True
    for group, ids in (("tune", tune), ("holdout", holdout)):
        want = expected_files(ids)
        for subdir in SUBDIRS:
            got = png_files(os.path.join(DST, group, subdir))
            ok = describe_difference(group, subdir, got, want) and ok
    return ok


def expected_manifest_rows(tune, holdout):
    rows = []
    rows.extend((sid, "tune", "阶段一选参") for sid in tune)
    rows.extend((sid, "holdout", "阶段二独立评测（不参与选参）") for sid in holdout)
    return rows


def check_manifest(tune, holdout):
    """检查仓库中的划分清单是否与固定规则完全一致。"""
    path = os.path.join(DST, "scene_split.csv")
    if not os.path.isfile(path):
        print("  [缺少] data_split/scene_split.csv")
        return False
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except (OSError, csv.Error) as exc:
        print("  [无法读取] data_split/scene_split.csv：%s" % exc)
        return False

    expected = [["场景编号", "分组", "用途"]]
    expected.extend([list(row) for row in expected_manifest_rows(tune, holdout)])
    if rows == expected:
        print("  [通过] data_split/scene_split.csv 共 200 个场景")
        return True

    print("  [不一致] data_split/scene_split.csv 与固定划分规则不一致")
    return False


def write_manifest(tune, holdout):
    os.makedirs(DST, exist_ok=True)
    path = os.path.join(DST, "scene_split.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["场景编号", "分组", "用途"])
        writer.writerows(expected_manifest_rows(tune, holdout))
    print("  已写清单 %s" % path)


def missing_source_files(tune, holdout):
    missing = []
    for _group, ids in (("tune", tune), ("holdout", holdout)):
        for subdir in SUBDIRS:
            for name in sorted(expected_files(ids)):
                path = os.path.join(SRC, subdir, name)
                if not os.path.isfile(path):
                    missing.append(path)
    return missing


def destination_extras(tune, holdout):
    """返回目标目录中不属于当前固定分组的 PNG，避免旧文件混入新划分。"""
    extras = []
    for group, ids in (("tune", tune), ("holdout", holdout)):
        want = expected_files(ids)
        for subdir in SUBDIRS:
            path = os.path.join(DST, group, subdir)
            got = png_files(path)
            if got is not None:
                extras.extend(os.path.join(path, name) for name in sorted(got - want))
    return extras


def copy_split(tune, holdout):
    for group, ids in (("tune", tune), ("holdout", holdout)):
        for subdir in SUBDIRS:
            os.makedirs(os.path.join(DST, group, subdir), exist_ok=True)
        for sid in ids:
            name = "%s_%s.png" % (sid, FRAME)
            for subdir in SUBDIRS:
                shutil.copy2(os.path.join(SRC, subdir, name),
                             os.path.join(DST, group, subdir, name))
        print("  已复制 %s：%d 个场景 × %d 个子目录" % (group, len(ids), len(SUBDIRS)))


def main():
    parser = argparse.ArgumentParser(description="固定划分选参组 150 + 验收组 50")
    parser.add_argument("--check", action="store_true", help="只核对已有划分，不复制或修改文件")
    args = parser.parse_args()

    tune, holdout = split_ids()
    print("划分规则：sorted(000000~000199) → random.Random(%d).shuffle → 前 %d 个 tune，末尾 %d 个 holdout"
          % (SEED, len(tune), len(holdout)))

    if args.check:
        ok = check_manifest(tune, holdout)
        ok = check_split(tune, holdout) and ok
        print("\n核对结果：%s" % ("全部通过" if ok else "存在问题"))
        return 0 if ok else 1

    missing = missing_source_files(tune, holdout)
    if missing:
        print("错误：原始数据缺少 %d 个必需文件，未开始复制。" % len(missing))
        for path in missing[:10]:
            print("  %s" % path)
        if len(missing) > 10:
            print("  ……其余 %d 个省略" % (len(missing) - 10))
        return 1

    extras = destination_extras(tune, holdout)
    if extras:
        print("错误：data_split 中有 %d 个不属于固定划分的 PNG，未继续复制。" % len(extras))
        for path in extras[:10]:
            print("  %s" % path)
        print("请先确认并移走这些旧文件，再重新运行脚本。")
        return 1

    copy_split(tune, holdout)
    write_manifest(tune, holdout)
    ok = check_split(tune, holdout) and check_manifest(tune, holdout)
    print("\n创建结果：%s" % ("全部通过" if ok else "存在问题"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
