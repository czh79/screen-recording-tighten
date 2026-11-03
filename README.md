# ✂️ 录屏智能去停顿 · Screen Recording Tightener

> 讲课时卡壳、喝水、翻笔记的那几秒，全给你剪掉——用神经网络识别人声边界，而不是靠音量阈值猜。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Whisper](https://img.shields.io/badge/faster--whisper-VAD-brightgreen)](https://github.com/SYSTRAN/faster-whisper)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**中文** | [English](#-english)

---

## 为什么按分贝剪不干净

用 `-25dB` 这种阈值找静音，是绝大多数教程的做法，问题是：

- 键盘敲击、鼠标点击、风扇底噪都会被当成"有人说话" → 剪不掉
- 一句话说到一半的换气被当成静音 → 剪出一个跳帧的鬼畜

这个脚本默认走 **faster-whisper 的 VAD（语音活动检测）**，它在神经网络层面判断"这一段是不是人声"，对噪声和短停顿的区分能力远超音量阈值。

## 两种模式

| 模式 | 原理 | 适合 |
|---|---|---|
| **1 · AI 智能跳切**（推荐） | Whisper VAD 神经网络识别人声区间 | 麦克风质量一般、环境有底噪 |
| **2 · 强力分贝切除** | `silencedetect` 滤镜 + 阈值 | 录音环境极干净、想剪得更狠 |

模式 1 偏保守（保留自然停顿），模式 2 偏激进（连呼吸气口一起切）。讲技术教程用 1，做快节奏口播用 2。

## 核心实现

```python
detect_speech_by_ai()   # VAD 拿到语音区间 → 合并相邻段 → 两端留 pad 缓冲
detect_speech_by_db()   # ffmpeg silencedetect 解析 → 反向求语音区间
process_tighten_video() # select/aselect 滤镜按区间重组，音频加人声增强
```

关键的几个细节：

- **`pad_sec = 0.12`** — 每段前后各留 120ms，避免切掉字头字尾的辅音
- **`min_silence_sec = 0.4`** — 短于 400ms 的停顿不剪，否则语流会碎
- **`>= 0.18s` 过滤** — 短于 180ms 的碎片直接丢弃，防止剪出闪烁帧
- **人声增强链** — `highpass + equalizer + compand`，剪完顺手把音质提一档
- **两阶段处理** — 先把每个语音区间用 `-ss/-t` 精确裁切（CRF 18 高质量编码 + 人声增强），再用 `concat -c copy` 无损拼接，拼接阶段不再二次转码

## 快速开始

```bash
git clone https://github.com/czh79/screen-recording-tighten.git
cd screen-recording-tighten
pip install -r requirements.txt
python tighten.py
```

脚本扫描当前目录下的视频（`.mp4/.mov/.mkv/.avi/.flv/.wmv/.m4v`），列出编号让你选。

```bash
# 指定别的目录
export VIDEO_INPUT_DIR="/path/to/recordings"
python tighten.py
```

输出：`tightened_<原名>.mp4`（模式 1）或 `tightened_strong_<原名>.mp4`（模式 2）。

## 效果参考

一条 25 分钟的课程录屏，通常能剪到 18–20 分钟——去掉的都是"呃…这个…""我找一下文件"这类真实无信息量的时间。

> 提示：剪完记得过一遍，AI VAD 偶发会把极轻的自言自语保留下来，手动补一刀即可。

## 配置项

| 变量 / 参数 | 默认 | 说明 |
|---|---|---|
| `WHISPER_MODEL_ZH` | `faster-whisper-small` | 中文识别模型（VAD 靠它） |
| `WHISPER_MODEL_EN` | `faster-whisper-base.en` | 英文模型 |
| `min_silence_sec` | `0.4` | 小于此值的停顿不剪 |
| `pad_sec` | `0.12` | 语音段两端留白 |
| `noise_db` | `-25dB` | 仅模式 2 使用 |

---

## 🇬🇧 English

> Remove dead air from screen recordings using neural VAD instead of a volume threshold.

Threshold-based silence removal (`-25dB`) fails in both directions: keyboard clicks and fan hum read as "speech" so they survive, while mid-sentence breaths read as silence and get cut, producing a stuttering jump-cut.

This tool uses faster-whisper's VAD to decide whether a span actually contains speech, then cuts each surviving range precisely (`-ss/-t`, CRF 18) with a `highpass + equalizer + compand` vocal chain, and finally stitches the pieces with `concat -c copy` — so the join step adds no extra generation loss.

### Two modes

- **AI jump-cut** (default) — VAD-driven, conservative, keeps natural rhythm.
- **Aggressive dB cut** — `silencedetect` based, also removes breaths; for fast-paced talking-head content.

### Usage

```bash
pip install -r requirements.txt
python tighten.py
```

Scans the current directory (override with `VIDEO_INPUT_DIR`), writes `tightened_<name>.mp4`.

Typical result: a 25-minute lecture lands around 18–20 minutes, with the removed time being genuine filler ("um… let me find that file").

---

<div align="center">

如果你也受够了剪停顿，**点个 Star** ⭐

</div>
