#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把原规则表 wordInfoRules.xlsx 导出成 gobehav 运行期使用的 conf/behav_<lang>.yaml。

四个 sheet 的去向：

    wordsInfo      -> words    词|词性|子类1..子类5（子类6 原实现未使用，丢弃）
    shortSentInfo  -> phrases  短语 -> 标签（保留 _S 后缀）
    rules          -> rules    检索规则；SDP/SRL 列原实现就是空列，DP 列见下方说明
    tagsPriority   -> tags     标签 -> id / 优先级 / 互斥优先级

关于 DP 列：原实现要靠 LTP 依存句法分析才能判定，gobehav 不带句法模型，
因此 DP 约束在导出时丢弃，规则退化为「忽略 DP」。受影响的规则会在 stderr 里列出来，
便于以后补一个轻量句法模块时回填。

用法：

    python tools/gobehav/export_assets.py
    python tools/gobehav/export_assets.py --lang cn --out conf/behav_cn.yaml

conf/ 是给人改的配置目录，不是纯生成物：已存在时默认不覆盖（免得冲掉手改的
规则），确实要重置成基线再加 --force。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import sys

try:
    import openpyxl
except ImportError:  # pragma: no cover
    sys.exit("需要 openpyxl：pip install openpyxl")

# 规则表列下标，与原 readExcel.py 一一对应
COL_ID, COL_CATEGORY, COL_TAG = 0, 1, 2
COL_SEARCH, COL_CHAPTER, COL_POS = 3, 4, 5
COL_DP, COL_SRL, COL_SDP, COL_LOCATION = 6, 7, 8, 9
COL_CONTEXT = 10  # 10..18 共 9 个上下文列

CONTEXT_KEYS = ["sub", "fa", "fda", "fca", "fdca", "la", "lda", "lca", "ldca"]

# 词性中文名 -> LTP 词性标记。规则表里 POS 列用的是 LTP 标记，
# 而 wordsInfo 用的是中文名，导出时统一成 LTP 标记后才能相互比较。
# 情感词/情绪词在 LTP 里没有对应标记（它们是内容词，按实际词性标注），
# 这里给一个占位标记，规则不会用到它们。
POS_MAP = {
    "动词": "v",
    "形容词": "a",
    "副词": "d",
    "介词": "p",
    "连词": "c",
    "代词": "r",
    "量词": "q",
    "数词": "m",
    "方位词": "nd",
    "标点": "wp",
    "情感词": "emo",
    "情绪词": "emo",
}


def sha256_of(path):
    """源表指纹，写进资产头部，便于核对导出结果对应的 xlsx 版本。"""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(row, idx):
    """取单元格并规整成去空白的字符串，None 视为空串。"""
    if idx >= len(row) or row[idx] is None:
        return ""
    val = row[idx]
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val).strip()


def split_multi(s):
    """按中英文逗号拆分多值单元格。"""
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,，]", s) if p.strip()]


def quote(s):
    """YAML 双引号转义，只对需要的字符做处理。"""
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def dump_tags(fh, rows):
    fh.write("tags:          # 标签 -> id / 优先级 p（越小越优先）/ 互斥优先级 x\n")
    for row in rows:
        tag = cell(row, 1)
        if not tag:
            continue
        tid = cell(row, 0) or "0"
        prio = cell(row, 2) or "4"
        exc = cell(row, 3)
        if exc:
            fh.write("  %s: {id: %s, p: %s, x: %s}\n" % (tag, tid, prio, exc))
        else:
            fh.write("  %s: {id: %s, p: %s}\n" % (tag, tid, prio))
    fh.write("\n")


def dump_phrases(fh, rows):
    fh.write("phrases:       # 短语 -> 标签列表（原 shortSentInfo，_S 表示短句标签优先）\n")

    # 同一个短语在不同「含义」下会挂多个标签（如 不好意思 -> 致歉 + 害羞），
    # 原实现把它们并到同一个位置再按优先级择一，这里保留全部标签交给引擎处理。
    grouped = {}
    order = []
    for row in rows:
        word = cell(row, 0)
        tag = cell(row, 2)
        if not word or not tag:
            continue
        keys = [t for t in re.split(r"[,，]", tag) if t]
        if word not in grouped:
            grouped[word] = []
            order.append(word)
        for key in keys:
            if key not in grouped[word]:
                grouped[word].append(key)

    for word in order:
        fh.write("  %s: %s\n" % (quote(word), ",".join(grouped[word])))
    fh.write("\n")


def dump_words(fh, rows):
    fh.write("words:         # 词|词性|子类1|子类2|子类3|子类4|子类5\n")
    for row in rows:
        word = cell(row, 0)
        if not word:
            continue
        pos = POS_MAP.get(cell(row, 2), "")
        attrs = [cell(row, i) for i in range(3, 8)]
        while attrs and not attrs[-1]:
            attrs.pop()
        parts = [word, pos] + attrs
        fh.write("  - %s\n" % quote("|".join(parts)))
    fh.write("\n")


def dump_rules(fh, rows):
    fh.write("rules:         # 检索规则：search 支持词表检索 / &词性=子类=属性 检索\n")
    dp_dropped = []

    for row in rows:
        if not any(row):
            continue
        rid = cell(row, COL_ID)
        tag = cell(row, COL_TAG)
        if not tag:
            continue

        fields = []
        if rid:
            fields.append("id: %s" % rid)
        fields.append("tag: %s" % tag)
        if cell(row, COL_CATEGORY):
            fields.append("cat: %s" % cell(row, COL_CATEGORY))

        search = re.sub(r"\s+", "", cell(row, COL_SEARCH))
        if search:
            fields.append("search: %s" % quote(search))
        if cell(row, COL_CHAPTER):
            fields.append("ch: %s" % quote(cell(row, COL_CHAPTER)))
        if cell(row, COL_POS):
            fields.append("pos: %s" % quote(cell(row, COL_POS)))
        if cell(row, COL_LOCATION):
            # 原表 loc 列存在 'E ' 这类带尾空格的脏数据，这里统一去空白
            fields.append("loc: %s" % quote(cell(row, COL_LOCATION)))

        if cell(row, COL_DP):
            dp_dropped.append((rid, tag, cell(row, COL_DP)))

        for i, key in enumerate(CONTEXT_KEYS):
            val = cell(row, COL_CONTEXT + i)
            if val:
                fields.append("%s: %s" % (key, quote(val)))

        fh.write("  - {%s}\n" % ", ".join(fields))

    fh.write("\n")
    return dp_dropped


# 规则表在 triton_inference_server 仓库里，而 gobehav 是独立仓库，
# 因此默认路径靠「向上逐级找同名兄弟目录」来确定，避免写死绝对路径。
XLSX_TAIL = os.path.join(
    "models", "behavior_emotion", "behavior_emotion_cn", "1",
    "cn_behavior_analysis", "data", "wordInfoRules.xlsx",
)


def find_xlsx(start):
    base = os.path.abspath(start)
    for _ in range(8):
        for cand in (os.path.join(base, "triton_inference_server"),
                     os.path.join(base, "hai", "triton_inference_server")):
            path = os.path.join(cand, XLSX_TAIL)
            if os.path.isfile(path):
                return path
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return ""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))

    default_xlsx = find_xlsx(root)

    ap = argparse.ArgumentParser(description="导出 gobehav 资产")
    ap.add_argument("--xlsx", default=default_xlsx, help="原规则表 wordInfoRules.xlsx")
    ap.add_argument("--lang", default="cn", help="语种，决定输出文件名 behav_<lang>.yaml")
    ap.add_argument("--out", default="", help="输出文件，默认 conf/behav_<lang>.yaml")
    ap.add_argument("--force", action="store_true",
                    help="输出文件已存在时也覆盖（默认不覆盖，保护手改过的规则）")
    args = ap.parse_args()

    xlsx = os.path.abspath(args.xlsx)
    if not os.path.isfile(xlsx):
        sys.exit("找不到规则表：%s\n用 --xlsx 指定路径。" % xlsx)

    out = args.out or os.path.join(root, "conf", "behav_%s.yaml" % args.lang)
    out = os.path.abspath(out)
    if os.path.exists(out) and not args.force:
        sys.exit("已存在，未覆盖：%s\n它是配置目录里的可改文件，确认要重置成基线请加 --force。" % out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    words = [r for r in wb["wordsInfo"].iter_rows(min_row=2, values_only=True) if cell(r, 0)]
    phrases = [r for r in wb["shortSentInfo"].iter_rows(min_row=2, values_only=True) if cell(r, 0)]
    tags = [r for r in wb["tagsPriority"].iter_rows(min_row=2, values_only=True) if cell(r, 1)]
    # rules 表前两行是两级表头
    rules = [r for r in wb["rules"].iter_rows(min_row=3, values_only=True) if any(r)]

    # 头部留痕：资产是生成物，必须能从这几行反推「谁、何时、从哪、用什么命令」导出，
    # 以及哪些原始字段被主动丢弃。改这里等于改溯源信息，请同步更新 README。
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 由 tools/gobehav/export_assets.py 从 wordInfoRules.xlsx 导出的基线。\n")
        fh.write("# conf/ 是配置目录，允许手改；重跑导出会提示，确认后可加 --force 重置。\n")
        fh.write("# 重新导出：python tools/gobehav/export_assets.py --xlsx <wordInfoRules.xlsx> --force\n")
        fh.write("# 导出时间：%s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fh.write("# 源文件：%s\n" % os.path.basename(xlsx))
        fh.write("# 源文件位置：%s\n" % XLSX_TAIL.replace(os.sep, "/"))
        fh.write("# 源文件 SHA256：%s\n" % sha256_of(xlsx))
        fh.write("# 规模：词 %d / 短语 %d / 标签 %d / 规则 %d\n"
                 % (len(words), len(phrases), len(tags), len(rules)))
        fh.write("# 未导出的原始列：SRL、SDP（原表整列为空）；"
                 "DP（依赖 LTP 依存句法，gobehav 无句法模型，含该约束的规则按「忽略 DP」导出）\n")
        fh.write("# 词性口径：原表中文词性已映射为 LTP 标记（动词->v / 副词->d / 介词->p / "
                 "连词->c / 代词->r / 量词->q / 数词->m / 方位词->nd / 标点->wp / 情感词·情绪词->emo）\n")
        fh.write("version: 1\n")
        fh.write("lang: %s\n\n" % args.lang)

        dump_tags(fh, tags)
        dump_phrases(fh, phrases)
        dump_words(fh, words)
        dp_dropped = dump_rules(fh, rules)

    size = os.path.getsize(out)
    print("已生成 %s" % out)
    print("  词 %d / 短语 %d / 标签 %d / 规则 %d / %.1f KB"
          % (len(words), len(phrases), len(tags), len(rules), size / 1024.0))

    if dp_dropped:
        print("\n以下 %d 条规则含 DP 约束，已按「忽略 DP」导出（原实现需 LTP 依存句法）："
              % len(dp_dropped))
        for rid, tag, dp in dp_dropped:
            print("  #%s %s dp=%s" % (rid, tag, dp))


if __name__ == "__main__":
    main()
