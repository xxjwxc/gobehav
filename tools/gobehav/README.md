# 资产导出

`conf/behav_cn.yaml` 不是手写的，是这份脚本从原 Triton 模型仓库的
`wordInfoRules.xlsx` 生成的。仓库里保留脚本 + 溯源信息，是为了以后
"规则改了怎么重出一份"有据可查，不用再翻旧代码反推字段含义。

## 数据来源

```
triton_inference_server/models/behavior_emotion/behavior_emotion_cn/
  1/cn_behavior_analysis/data/wordInfoRules.xlsx
```

四个 sheet 的去向：

| sheet | 产物节点 | 说明 |
| --- | --- | --- |
| `wordsInfo` | `words` | 词 `\|` 词性 `\|` 子类1..5；子类6 原实现未使用，丢弃 |
| `shortSentInfo` | `phrases` | 短语 -> 标签列表，`_S` 后缀表示短句标签 |
| `rules` | `rules` | 前两行为两级表头，数据从第 3 行开始 |
| `tagsPriority` | `tags` | 标签 -> `id` / `p`（优先级）/ `x`（互斥优先级） |

列下标与原 `readExcel.py` 一一对应，见脚本顶部 `COL_*` 常量。

同级目录下还有 6 份多语种表（`wordInfoRules_hi/de/en/es/fr/it.xlsx`），
但它们只有一行表头、没有数据，对应链路真正依赖的是未随仓库提交的 ONNX 情感分类模型，
因此没有导出，详见根目录 README 的「其他语种为什么不在」。

## 重新导出

```bash
python -m venv .venv
.venv/bin/pip install -r tools/gobehav/requirements.txt
.venv/bin/python tools/gobehav/export_assets.py --xlsx /path/to/wordInfoRules.xlsx
```

不带 `--xlsx` 时，脚本从仓库根向上逐级查找 `triton_inference_server/.../wordInfoRules.xlsx`，
找到即可直接跑。输出路径默认 `conf/behav_<lang>.yaml`，可用 `--out` 覆盖。

导出结束后校验：

```bash
go test ./...
```

## 溯源信息写在哪里

生成的 YAML 头部会写入：重新导出命令、导出时间、源文件名与相对位置、
源文件 SHA256、四类资产规模、未导出的原始列、词性映射口径。
所以拿到一份资产就能反推它是从哪版 xlsx 出来的。

改了脚本里的映射规则，务必同步更新 `README.md` 的「资产与导出」章节。

## 被裁掉的字段

| 列 | 原因 |
| --- | --- |
| `SRL`、`SDP` | 原表整列为空，原实现里也是死配置 |
| `DP` | 依赖 LTP 依存句法（SBV/VOB/POB/LAD…），gobehav 不带句法模型。含该约束的规则按"忽略 DP"导出，脚本会在 stderr 列出全部 16 条，便于以后补轻量句法模块时回填 |

## 词性口径

原表 `wordsInfo` 的词性是中文名，规则表 `pos` 列是 LTP 标记，两边不在一个命名空间。
导出时统一成 LTP 标记：

```
动词->v  副词->d  介词->p  连词->c  代词->r  量词->q
数词->m  方位词->nd  标点->wp  情感词->emo  情绪词->emo
```

`emo` 是占位标记：情感词/情绪词在 LTP 里按实际词性标注，规则不会用这两个名字做约束。

---

# bge-m3 语义兜底（可选）

规则引擎的上限是词表：词表里没有的说法一条标签也打不出。bge-m3 只在
**规则层完全空命中**时补一个语义最近的标签，不参与正常路径。

## 为什么不自己导 fp32

实测 ModelScope 上 `BAAI/bge-m3` 仓库已带官方 ONNX：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `onnx/model.onnx` + `onnx/model.onnx_data` | 725 KB + 2.27 GB | fp32，external-data 格式 |
| `onnx/tokenizer.json` | 17 MB | 与图配套的分词文件 |
| `1_Pooling/config.json` | - | `pooling_mode_cls_token: true` |

图输入 `input_ids`/`attention_mask`（int64，batch 与 seq 均动态），输出
`token_embeddings` 与已 L2 归一化的 `sentence_embedding`。实测
`torch normalize(last_hidden_state[:,0])` 与 `sentence_embedding` 余弦 = **1.0**，
所以直接复用官方图，不再用 torch 重复导一遍。

HuggingFace 在内网不通（`curl https://huggingface.co` 超时），权重从 ModelScope 拉。

## 命令

```bash
# 1. 拉权重（约 4.3 GB，含 pytorch bin 与 onnx）
.venv/bin/modelscope download --model BAAI/bge-m3 --local_dir ~/models/bge-m3

# 2. 量化 + 校验 + 延迟 + 标签原型
.venv/bin/python tools/gobehav/export_bge_m3.py \
    --model ~/models/bge-m3 \
    --outdir assets/bge_m3 \
    --behav-yaml conf/behav_cn.yaml

# 3. 评兜底质量（原型 vs kNN），会额外落盘 phrase_vectors.f32
.venv/bin/python tools/gobehav/eval_bge.py
```

依赖见 `requirements-bge.txt`。GPU 延迟实测没做，原因见下方实测结果表。

## 产物

| 文件 | 大小 | 是否入库 |
| --- | --- | --- |
| `encoder.int8.onnx` + `.data` | 1.0 MB + 568 MB | 否（`.gitignore`） |
| `tokenizer.json` | 17 MB | 否（`.gitignore`） |
| `label_prototypes.f32` | 292 KB | 是 |
| `label_index.json` | 3.6 KB | 是 |
| `meta.json` | 3.7 KB | 是 |
| `phrase_vectors.f32` + `phrase_index.json` | 12.3 MB + 160 KB | 否（可选 kNN 产物） |

## 实测结果（2026-11-05，RTX 3090 机器 / 16 核 / Python 3.12）

| 项 | 结果 |
| --- | --- |
| fp32 vs torch | 余弦 1.0（精确一致） |
| INT8 vs fp32 | min 0.98262 / mean 0.98555，最大绝对差 0.0228 |
| INT8 体积 | 568 MB（原 fp32 2.27 GB，压到 1/4） |
| CPU 单条（seq=10） | 14.7 ms |
| CPU 批量 8 条 | 56.6 ms（7.1 ms/条） |
| GPU | **未测** |

> 3090 在机器上但没跑成 GPU：内网到 pypi 镜像约 16 MB/min，ORT 的 CUDA
> 依赖（`onnxruntime-gpu` 256 MB + cuDNN 716 MB + cuBLAS）合计 1.1 GB，
> 单 cuDNN 就要 45 分钟，性价比不足。而且兜底走的是 Go 服务侧 CPU，
> CPU 数字才是实际部署口径。

## 兜底质量

拿 3152 条短语回查自己的标签（`eval_bge.py`），比两种兜底方案：

| 方案 | 体积 | top1 | top3 | @0.62 覆盖/精度 | @0.75 覆盖/精度 |
| --- | --- | --- | --- | --- | --- |
| 标签原型（73×1024 类均值） | 292 KB | 0.6421 | **0.8886** | 0.763 / 0.685 | 0.221 / 0.842 |
| 短语 kNN（3152×1024） | 12.3 MB | 0.6466 | 0.8052 | **0.869 / 0.693** | 0.493 / 0.805 |

入库选**原型**：体积小 40 倍，top3 反而高 8 个点；kNN 只在“高覆盖率 + 高阈值”
区间有优势，属于可选。

**top1 只有 64% 不是模型不行，是标签表粒度重叠。**错判排行里几乎全是相邻标签：
`mood_happy` ↔ `mood_satisfied`、`behavior_no` ↔ `behavior_reject`、
`mood_nervous` ↔ `mood_panic`、`mood_sad` ↔ `mood_dissapointment`。
余弦 ≥ 0.90 的短语对共 253 组，其中标签完全不重合的只有 12 组（4.7%），
说明数据本身不脏，是“两个标签本来就是一个意思”。

所以兜底定位是**弱提示**：只给 1 个标签 + 分数，调用方自己决定要不要用。

## 阈值

| 阈值 | 短语覆盖率 | 覆盖内 top1 精度 |
| --- | --- | --- |
| 0.60 | 0.829 | 0.678 |
| 0.62 | 0.763 | 0.685 |
| 0.65 | 0.658 | 0.706 |
| 0.70 | 0.420 | 0.750 |
| 0.75 | 0.221 | 0.842 |

默认 **0.62**：再低就开始乱标；要更高精度就抬到 0.70（精度 0.75，代价是覆盖率
从 76% 掉到 42%）。词表外句子实测 top1 分数在 0.63~0.75，刚好在阈值附近。

## Go 侧验证（`bge` 子包）

`bge` 子包把上面三个文件接成 `gobehav.Fallback`。在内网 16 核机器上跑通，
并与 Python 侧逐条对拍：

| 项 | 结果 |
| --- | --- |
| 分词 vs transformers | 12 条样例 `input_ids` 逐位相同（含中英混排、多余空格、空串） |
| 向量 vs Python onnxruntime | 前 16 维最大绝对差 **0**，余弦 1.0 |
| Go 单条（分词+推理+原型） | 14.47 ms |
| Go 批量 8 条 | 54.52 ms（6.82 ms/条） |
| 二进制体积 | 9.0 MB |

libonnxruntime 不用另外装，直接用 venv 里现成的：

```
~/bge-m3-venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1
```

分词器没走 Rust 绑定：`daulet/tokenizers` 的 Go module 已经不再自带
`libtokenizers.a`，内网编不了；改成用 `sugarme/tokenizer` 的零件（Precompiled +
WhiteSpaceSplit/Metaspace + Unigram）手工组装，纯 Go，结果与 transformers 一致。

## 标签原型

把 `behav_cn.yaml` 的短语按标签聚成桶，每个标签的短语向量取均值再归一化，
得到 `[73, 1024]` 的原型矩阵（只有 73 个标签挂短语，其余 67 个只能靠规则）。

用词表外句子测区分度：

| 输入 | top1 | 分数 |
| --- | --- | --- |
| 我真的栓Q了 | `mood_helpless` | 0.6532 |
| 这波操作我是真的服了 | `behavior_thanking` | 0.7056 |
| 这事儿整挺好 | `behavior_yes` | 0.7468 |
| 又emo了，不想说话 | `behavior_hush` | 0.7384 |
| 绝绝子，太过分了吧 | `mood_blame` | 0.6346 |

区分度在这个区间可用，但 **0.62 以下建议直接放弃兜底**：再低就开始乱标。

上面 5 条的分数都落在 0.63~0.75，正好说明 0.62 这个阈值是贴着分布下沿定的：
能拦住明显无关的句子，但宁可漏、不错标。

