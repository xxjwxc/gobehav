"""评估语义兜底的两种做法：标签原型（类均值）vs 短语 kNN（类内最近邻）。

只在导出机器上跑，产物是打印出来的数字（写进 tools/gobehav/README.md）。
    python eval_bge.py [--limit N]
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer

MD = "/home/niren/models/bge-m3"
OUT = "/home/niren/gobehav_out/bge_m3"
YAML = "/home/niren/gobehav_tools/behav_cn.yaml"
MAX_LEN = 64
BATCH = 16
THRESHOLDS = [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75]


def base_tag(t: str) -> str:
    """短语表的标签带 `_S`（短句优先）后缀，原型矩阵的标签不带。"""
    t = t.strip()
    return t[:-2] if t.endswith("_S") else t


def load_prototypes():
    idx = json.load(open(os.path.join(OUT, "label_index.json"), encoding="utf-8"))
    tags, dim = idx["tags"], idx["dim"]
    mat = np.fromfile(os.path.join(OUT, "label_prototypes.f32"), dtype=np.float32)
    return tags, mat.reshape(len(tags), dim)


def load_phrases():
    doc = yaml.safe_load(open(YAML, encoding="utf-8"))
    out = []
    for phrase, raw in doc["phrases"].items():
        tags = {base_tag(x) for x in str(raw).split(",") if x.strip()}
        out.append((phrase, tags))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条短语，0 表示全量")
    args = ap.parse_args()

    tag_names, proto = load_prototypes()
    pos = {t: i for i, t in enumerate(tag_names)}
    phrases = load_phrases()
    if args.limit:
        phrases = phrases[: args.limit]

    # 短语可能带原型库没有的标签（没有短语的标签凑不出原型），剔除掉
    phrases = [(p, {t for t in ts if t in pos}) for p, ts in phrases]
    phrases = [(p, ts) for p, ts in phrases if ts]
    print("评估短语数: %d / 原型标签数: %d / 维度: %d" % (len(phrases), len(tag_names), proto.shape[1]))

    tok = AutoTokenizer.from_pretrained(MD)
    sess = ort.InferenceSession(
        os.path.join(OUT, "encoder.int8.onnx"), providers=["CPUExecutionProvider"]
    )

    rows = []
    texts = [p for p, _ in phrases]
    all_vec = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start: start + BATCH]
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        vec = sess.run(["sentence_embedding"], feed)[0]
        all_vec.append(vec)
        sims = vec @ proto.T                       # [b, n_tag]
        order = np.argsort(-sims, axis=1)[:, :3]   # 只看 top3
        for i, phrase in enumerate(chunk):
            gold = phrases[start + i][1]
            top = [(tag_names[j], float(sims[i, j])) for j in order[i]]
            rows.append((phrase, gold, top))

    report("标签原型（类均值）", rows)

    # ---- 方案二：短语 kNN，直接用类内最近的那条短语投票 ----
    vecs = np.concatenate(all_vec, axis=0)
    vecs.tofile(os.path.join(OUT, "phrase_vectors.f32"))
    with open(os.path.join(OUT, "phrase_index.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "dim": int(vecs.shape[1]),
            "count": int(vecs.shape[0]),
            "dtype": "float32",
            "layout": "row-major [count, dim], L2 normalized",
            "items": [
                {"text": p, "tags": sorted(g)}
                for (p, _), g in zip(phrases, [g for _, g in phrases])
            ],
        }, fp, ensure_ascii=False)
    print("\n短语向量已落盘: phrase_vectors.f32 %.1f MB / phrase_index.json"
          % (vecs.nbytes / 1048576))

    knn = vecs @ vecs.T
    np.fill_diagonal(knn, -1.0)                    # 排掉自己
    order = np.argsort(-knn, axis=1)[:, :3]
    krows = []
    for i, (phrase, gold) in enumerate(phrases):
        top = []
        for j in order[i]:
            # 邻居短语可能挂多个标签，取相似度最高的那个标签参与比较
            tag = sorted(phrases[j][1])[0]
            top.append((tag, float(knn[i, j])))
        krows.append((phrase, gold, top))
    report("短语 kNN（类内最近邻）", krows)

    # 上面两类错判里有一大堆是“同义短语被赋了不同标签”
    # （心如不定->mood_nervous 而 心如不宁->mood_panic），是源数据自身的噪声。
    # 这里把“近义短语的标签”也算进 gold，量一下噪声占多少。
    golds = [g for _, g in phrases]
    same = knn >= 0.90
    noisy = 0
    pairs = 0
    for i in range(len(phrases)):
        for j in np.nonzero(same[i])[0]:
            if j <= i:
                continue
            pairs += 1
            if not (golds[i] & golds[j]):
                noisy += 1
    print("\n疑似标签噪声：余弦>=0.90 的短语对 %d 个，其中标签完全不重合 %d 个（%.1f%%）"
          % (pairs, noisy, 100.0 * noisy / max(pairs, 1)))

    relaxed = []
    for i, (phrase, gold) in enumerate(phrases):
        near = set(gold)
        for j in np.nonzero(knn[i] >= 0.95)[0]:
            near |= golds[j]
        top = [(sorted(golds[j])[0], float(knn[i, j])) for j in order[i]]
        relaxed.append((phrase, near, top))
    report("短语 kNN（放宽 gold：合并余弦>=0.95 近义短语的标签）", relaxed)



def report(name, rows):
    print("\n=== %s ===" % name)
    top1 = sum(1 for _, g, t in rows if t[0][0] in g) / len(rows)
    top3 = sum(1 for _, g, t in rows if any(n in g for n, _ in t)) / len(rows)
    print("top1 命中: %.4f   top3 命中: %.4f" % (top1, top3))

    print("\n阈值 -> 覆盖率 / 覆盖率内 top1 精度")
    for thr in THRESHOLDS:
        kept = [(g, t) for _, g, t in rows if t[0][1] >= thr]
        if not kept:
            print("  %.2f  coverage=0.0000" % thr)
            continue
        acc = sum(1 for g, t in kept if t[0][0] in g) / len(kept)
        print("  %.2f  coverage=%.4f  precision=%.4f  n=%d"
              % (thr, len(kept) / len(rows), acc, len(kept)))

    bad = [(t[0][1], p, t[0][0], next(iter(g))) for p, g, t in rows if t[0][0] not in g]
    bad.sort(reverse=True)
    print("\n最自信的 10 个错判（score 降序）:")
    for score, phrase, pred, gold in bad[:10]:
        print("  %.4f  %s  -> %s (期望 %s)" % (score, phrase, pred, gold))


if __name__ == "__main__":
    main()
