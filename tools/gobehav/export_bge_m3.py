#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bge-m3 → gobehav 兜底向量：量化、校验、延迟实测、标签原型库。

为什么不再自己导 fp32 ONNX：ModelScope 上的 `BAAI/bge-m3` 仓库已带
`onnx/model.onnx` + `model.onnx_data`，输入 `input_ids`/`attention_mask`，
输出 `token_embeddings` 与已 L2 归一化的 `sentence_embedding`（CLS pooling），
与 PyTorch 的 `normalize(last_hidden_state[:,0])` 余弦 = 1.0。
本脚本直接复用官方 fp32 图，只做官方没提供的三件事：

    1. INT8 动态量化（CPU 上把权重压到 1/4，模型从 2.3G 降到 ~600M）
    2. 一致性校验：int8 vs fp32 的逐条余弦
    3. 延迟实测：CPU / GPU（onnxruntime-gpu 存在时）单条与批量
    4. 可选：用 behav_cn.yaml 的短语算 140 个标签的原型向量

产物（--outdir）：

    encoder.int8.onnx[.data]   INT8 量化模型
    tokenizer.json             与模型配套的分词文件（Go 侧解析用）
    label_prototypes.f32       标签原型向量，float32 小端，行优先 [n_tags, 1024]
    label_index.json           标签顺序与阈值，与上面二进制一一对应
    meta.json                  溯源：来源、时间、版本、sha256、校验结果、延迟

用法：

    python tools/gobehav/export_bge_m3.py \
        --model ~/models/bge-m3 \
        --outdir assets/bge_m3 \
        --behav-yaml conf/behav_cn.yaml

    # 只量化不测延迟
    python tools/gobehav/export_bge_m3.py --model ~/models/bge-m3 --no-bench
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

SAMPLE_TEXTS = [
    "对不起，我真的很抱歉",
    "我今天特别开心，哈哈哈哈",
    "这个多少钱？",
    "虽然很难，但我还是要坚持",
    "请帮我一下，麻烦你了",
    "我不喜欢这个",
    "谢谢你，辛苦了",
    "整个都不好了",
]

# 测阈值的“词表外”句子：规则层基本打不出标签，正好用来看向量兜底的区分度
PROBE_TEXTS = [
    "这波操作我是真的服了",
    "绝绝子，太过分了吧",
    "我真的栓Q了",
    "又emo了，不想说话",
    "这事儿整挺好",
]


def base_tag(t: str) -> str:
    """短语表的标签带 `_S`（短句优先）后缀，规则标签不带；归并到基础名上。"""
    t = t.strip()
    return t[:-2] if t.endswith("_S") else t


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for blk in iter(lambda: fp.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f GB" % size


def encode(tok, texts, max_len):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="np")
    return {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}


def pick_output(sess) -> str:
    """优先取已池化归一化的 sentence_embedding，没有就退回 token_embeddings。"""
    names = [o.name for o in sess.get_outputs()]
    if "sentence_embedding" in names:
        return "sentence_embedding"
    return names[0]


def embed(sess, feed):
    name = pick_output(sess)
    out = sess.run([name], feed)[0]
    if out.ndim == 3:  # [b, s, h] -> 取 CLS
        out = out[:, 0]
    return out


def quantize(fp32_path: str, int8_path: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # bge-m3 是 2.3G 的 external-data 模型，必须让 ORT 以外部数据方式落盘，
    # 否则会撞上 protobuf 2GB 上限。
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=True,
    )


def compare(tok, ref_sess, tgt_sess, texts, max_len):
    feed = encode(tok, texts, max_len)
    ref = embed(ref_sess, feed)
    got = embed(tgt_sess, feed)
    cos = np.sum(ref * got, axis=1) / (
        np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-12)
    return {
        "min_cos": round(float(cos.min()), 6),
        "mean_cos": round(float(cos.mean()), 6),
        "max_abs_diff": round(float(np.abs(ref - got).max()), 6),
    }


def bench(sess, tok, texts, max_len, rounds=20):
    feed = encode(tok, texts, max_len)
    single = {k: v[:1] for k, v in feed.items()}

    for _ in range(3):  # 预热
        embed(sess, single)
        embed(sess, feed)

    t0 = time.perf_counter()
    for _ in range(rounds):
        embed(sess, single)
    single_ms = (time.perf_counter() - t0) * 1000 / rounds

    t0 = time.perf_counter()
    for _ in range(rounds):
        embed(sess, feed)
    batch_ms = (time.perf_counter() - t0) * 1000 / rounds

    return {
        "providers": sess.get_providers(),
        "batch_size": len(texts),
        "seq_len_max": int(feed["input_ids"].shape[1]),
        "single_ms": round(single_ms, 2),
        "batch_ms": round(batch_ms, 2),
        "per_item_ms": round(batch_ms / len(texts), 2),
    }


def build_prototypes(sess, tok, yaml_path, max_len, outdir):
    """每个标签的原型 = 该标签下所有短语向量的均值，再 L2 归一化。"""
    import yaml

    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    phrases = data.get("phrases") or {}
    tags = data.get("tags") or {}

    buckets: dict[str, list[str]] = {}
    for phrase, val in phrases.items():
        if not phrase or not val:
            continue
        for t in str(val).split(","):
            t = base_tag(t)
            if t:
                buckets.setdefault(t, []).append(phrase)

    # 只保留标签表里存在的标签，顺序固定，便于 Go 侧按下标取。
    order = [t for t in tags if t in buckets]
    if not order:
        print("  behav yaml 里没有可用的短语，跳过原型向库")
        return None

    uniq = sorted({p for t in order for p in buckets[t]})
    print("  标签 %d 个 / 唯一短语 %d 条，开始编码 ..." % (len(order), len(uniq)))

    vecs = {}
    step = 32
    t0 = time.time()
    for i in range(0, len(uniq), step):
        chunk = uniq[i:i + step]
        out = embed(sess, encode(tok, chunk, max_len))
        for p, v in zip(chunk, out):
            vecs[p] = v
        if (i // step) % 10 == 0:
            print("    %d/%d  %.0fs" % (min(i + step, len(uniq)), len(uniq), time.time() - t0))

    dim = next(iter(vecs.values())).shape[0]
    mat = np.zeros((len(order), dim), dtype=np.float32)
    for i, t in enumerate(order):
        m = np.mean([vecs[p] for p in buckets[t]], axis=0)
        mat[i] = m / (np.linalg.norm(m) + 1e-12)

    bin_path = os.path.join(outdir, "label_prototypes.f32")
    mat.tofile(bin_path)

    index = {
        "source": os.path.basename(yaml_path),
        "dim": int(dim),
        "count": len(order),
        "dtype": "float32",
        "layout": "row-major [count, dim], L2 normalized",
        "tags": order,
        "phrases_per_tag": {t: len(buckets[t]) for t in order},
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    idx_path = os.path.join(outdir, "label_index.json")
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("  已写入 %s (%s) 与 %s" % (bin_path, human(os.path.getsize(bin_path)), idx_path))
    return {"file": os.path.basename(bin_path), "count": len(order), "dim": int(dim),
            "sha256": sha256_of(bin_path), "unique_phrases": len(uniq)}


def probe_prototypes(sess, tok, outdir, max_len, topk=3):
    """用词表外的句子看兜底效果：打印 top-k 相似标签与分数，便于定阈值。"""
    with open(os.path.join(outdir, "label_index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    mat = np.fromfile(os.path.join(outdir, "label_prototypes.f32"),
                      dtype=np.float32).reshape(idx["count"], idx["dim"])

    vecs = embed(sess, encode(tok, PROBE_TEXTS, max_len))
    sims = vecs @ mat.T  # 两边都已归一化，点积即余弦

    out = []
    for i, text in enumerate(PROBE_TEXTS):
        top = np.argsort(-sims[i])[:topk]
        out.append({
            "text": text,
            "top": [{"tag": idx["tags"][j], "score": round(float(sims[i][j]), 4)} for j in top],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="bge-m3 量化 / 校验 / 延迟 / 标签原型")
    ap.add_argument("--model", default="~/models/bge-m3", help="bge-m3 模型目录（含 onnx/）")
    ap.add_argument("--outdir", default="assets/bge_m3")
    ap.add_argument("--max-len", type=int, default=64, help="校验/实测用的最大长度")
    ap.add_argument("--behav-yaml", default="", help="behav_cn.yaml，用来生成标签原型")
    ap.add_argument("--with-gpu", action="store_true", help="额外的 GPU 延迟（需 onnxruntime-gpu）")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--force-quant", action="store_true", help="忽略已有的 int8，重新量化")
    ap.add_argument("--no-prototypes", action="store_true")
    args = ap.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model))
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    fp32_path = os.path.join(model_dir, "onnx", "model.onnx")
    if not os.path.isfile(fp32_path):
        sys.exit("找不到 %s；请先 modelscope download BAAI/bge-m3" % fp32_path)

    int8_path = os.path.join(outdir, "encoder.int8.onnx")

    print("模型目录 %s" % model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.save_pretrained(outdir)
    if not os.path.isfile(os.path.join(outdir, "tokenizer.json")) and getattr(tok, "is_fast", False):
        tok.backend_tokenizer.save(os.path.join(outdir, "tokenizer.json"))
    print("tokenizer 已保存到 %s" % outdir)

    meta = {
        "model": "BAAI/bge-m3",
        "source_onnx": "onnx/model.onnx (ModelScope 官方)",
        "pooling": "cls",
        "normalize": True,
        "dim": 1024,
        "max_len": args.max_len,
    }

    fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    meta["fp32"] = {"bytes": os.path.getsize(fp32_path),
                    "sha256": sha256_of(fp32_path),
                    "outputs": [o.name for o in fp32.get_outputs()]}

    # fp32 与 PyTorch 的对照，证明官方图和 CLS+normalize 口径一致
    print("对照 PyTorch ...")
    model = AutoModel.from_pretrained(model_dir).eval()
    feed = encode(tok, SAMPLE_TEXTS, args.max_len)
    with torch.no_grad():
        ref = model(input_ids=torch.from_numpy(feed["input_ids"]),
                    attention_mask=torch.from_numpy(feed["attention_mask"]))
        ref = torch.nn.functional.normalize(ref.last_hidden_state[:, 0], p=2, dim=1).numpy()
    got = embed(fp32, feed)
    meta["verify_torch"] = {
        "min_cos": round(float(np.min(np.sum(ref * got, axis=1))), 6),
        "note": "torch normalize(last_hidden_state[:,0]) vs onnx sentence_embedding",
    }
    print("  %s" % meta["verify_torch"])
    del model

    if not args.no_quant:
        if os.path.isfile(int8_path) and not args.force_quant:
            print("复用已有的 int8：%s（重新量化加 --force-quant）" % int8_path)
        else:
            print("INT8 动态量化 ...")
            t0 = time.time()
            quantize(fp32_path, int8_path)
            print("  %s  用时 %.1fs" % (human(os.path.getsize(int8_path)), time.time() - t0))
        meta["int8"] = {"file": os.path.basename(int8_path),
                        "bytes": os.path.getsize(int8_path),
                        "sha256": sha256_of(int8_path)}
        data = int8_path + ".data"
        if os.path.isfile(data):
            meta["int8"]["external_data"] = {"file": os.path.basename(data),
                                             "bytes": os.path.getsize(data)}

        int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        meta["verify_int8"] = compare(tok, fp32, int8, SAMPLE_TEXTS, args.max_len)
        print("  int8 vs fp32: %s" % meta["verify_int8"])

        # 逐条分数，便于人工判断兜底阈值
        f32v = embed(fp32, feed)
        i8v = embed(int8, feed)
        meta["sample_scores"] = [
            {"text": t,
             "cos_fp32_int8": round(float(np.dot(f32v[i], i8v[i])), 6)}
            for i, t in enumerate(SAMPLE_TEXTS)
        ]

        if args.behav_yaml and not args.no_prototypes:
            print("生成标签原型 ...")
            proto = build_prototypes(int8, tok, args.behav_yaml, args.max_len, outdir)
            meta["prototypes"] = proto
            if proto:
                meta["probe_scores"] = probe_prototypes(int8, tok, outdir, args.max_len)
                print("  词表外样例 top3: %s" % json.dumps(meta["probe_scores"], ensure_ascii=False))
    else:
        int8 = fp32

    if not args.no_bench:
        print("延迟实测 ...")
        meta["bench_cpu"] = bench(int8, tok, SAMPLE_TEXTS, args.max_len)
        print("  cpu: %s" % meta["bench_cpu"])
        if args.with_gpu:
            try:
                gpu = ort.InferenceSession(int8_path,
                                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                meta["bench_gpu"] = bench(gpu, tok, SAMPLE_TEXTS, args.max_len)
                print("  gpu: %s" % meta["bench_gpu"])
            except Exception as exc:  # noqa: BLE001
                meta["bench_gpu"] = {"error": str(exc)}
                print("  gpu 不可用: %s" % exc)

    meta["exported_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["versions"] = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    import transformers
    meta["versions"]["transformers"] = transformers.__version__

    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("已写入 %s/meta.json" % outdir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
