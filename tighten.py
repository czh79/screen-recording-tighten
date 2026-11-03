import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 统一指向多语言模型目录（用于完美支持中文及中英夹杂教程）
LOCAL_MODEL_ZH = os.environ.get("WHISPER_MODEL_ZH", "faster-whisper-small")
LOCAL_MODEL_EN = os.environ.get("WHISPER_MODEL_EN", "faster-whisper-base.en")

def extract_numbers_key(filepath: Path):
    stem = filepath.stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def scan_target_videos(input_dir: str):
    dir_path = Path(input_dir)
    if not dir_path.exists():
        print(f"❌ 目录不存在: {input_dir}")
        return []

    video_extensions = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".m4v"}
    return sorted([
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
        and not f.name.startswith("temp_")
        and not f.name.startswith("enhanced_")
        and not f.name.startswith("tightened_")
    ], key=extract_numbers_key)


def get_video_duration(video_file: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_file.resolve())
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except FileNotFoundError:
        pass
    return 0.0


def get_vocal_enhancement_filter() -> str:
    """优化后的播音级人声滤镜：解决声音发闷，提升清亮感与空气感"""
    filters = [
        "highpass=f=60",                                      # 降低高切起点（从85Hz降到60Hz，保留更多人声低频磁性，不发闷）
        "lowpass=f=14000",                                    # 适当提高高通上限，保留更多高频细节
        "afftdn=nf=-20",                                      # 减弱降噪强度（从-25调到-20），防止过度降噪吃掉人声细节
        "agate=threshold=0.03:ratio=2.5:range=0.12:attack=15:release=100",  # 气口与呼吸声抑制
        "equalizer=f=4000:t=q:w=1.2:g=3.5",                   # 将咬字提亮频段上移至 4kHz 并稍微加大增益，让声音更清亮有穿透力
        "treble=g=2.0",                                       # 额外增加一层高频“空气感”（Air），彻底告别沉闷
        "loudnorm=I=-16:TP=-1.5:LRA=11",                      # 广播级响度平衡
    ]
    return ",".join(filters)


def detect_speech_by_ai(video_file: Path, min_silence_sec: float = 0.4, pad_sec: float = 0.12):
    """专门针对中文优化的 AI 人声探测"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("⚠️ 未检测到 faster-whisper，自动切换至分贝检测...")
        return None

    model_src = LOCAL_MODEL_ZH if os.path.exists(LOCAL_MODEL_ZH) else "small"
    print(f"🤖 [AI 中文人声探测] 正在加载多语言模型 ({model_src})...")

    try:
        model = WhisperModel(model_src, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(video_file.resolve()),
            language="zh",  # 明确指定中文普通话
            initial_prompt="以下是普通话教程句子，包含标点。",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=int(min_silence_sec * 1000),
                speech_pad_ms=int(pad_sec * 1000),
            ),
            word_timestamps=True,
        )

        raw_intervals = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    s = max(0.0, w.start - pad_sec)
                    e = w.end + pad_sec
                    raw_intervals.append((s, e))
            else:
                s = max(0.0, seg.start - pad_sec)
                e = seg.end + pad_sec
                raw_intervals.append((s, e))

        if not raw_intervals:
            return None

        merged = [raw_intervals[0]]
        for cur_s, cur_e in raw_intervals[1:]:
            prev_s, prev_e = merged[-1]
            if cur_s - prev_e < min_silence_sec:
                merged[-1] = (prev_s, max(prev_e, cur_e))
            else:
                merged.append((cur_s, cur_e))

        return merged
    except Exception as e:
        print(f"⚠️ 中文 AI 探测异常 ({e})，切换至分贝探测...")
        return None


def detect_speech_by_db(video_file: Path, total_dur: float, noise_db: str = "-25dB", min_dur: float = 0.4):
    print(f"📊 [分贝探测] 正在以 {noise_db} 阈值扫描停顿空白...")
    af_cmd = f"highpass=f=100,lowpass=f=8000,silencedetect=noise={noise_db}:d={min_dur}"
    cmd = ["ffmpeg", "-i", str(video_file.resolve()), "-af", af_cmd, "-f", "null", "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

    silence_starts = []
    silence_ends = []
    for line in res.stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([\d\.]+)", line)
            if m:
                silence_starts.append(float(m.group(1)))
        elif "silence_end:" in line:
            m = re.search(r"silence_end:\s*([\d\.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))

    if len(silence_starts) > len(silence_ends):
        silence_ends.append(total_dur)

    speech_ranges = []
    curr = 0.0
    pad = 0.10
    for s_start, s_end in zip(silence_starts, silence_ends):
        start_cut = max(0.0, s_start + pad)
        end_cut = max(start_cut, s_end - pad)
        if start_cut - curr >= 0.20:
            speech_ranges.append((curr, start_cut))
        curr = end_cut

    if total_dur - curr >= 0.20:
        speech_ranges.append((curr, total_dur))

    return speech_ranges


def process_tighten_video(input_file: Path, output_file: Path, use_ai_vad: bool = True, noise_db: str = "-25dB"):
    total_dur = get_video_duration(input_file)
    if total_dur <= 0:
        print(f"❌ 无法读取视频时长: {input_file.name}")
        return False

    print("\n" + "=" * 60)
    print(f"🎬 正在处理中文视频: {input_file.name} | 原始总时长: {total_dur:.2f} 秒")
    print("=" * 60)

    speech_ranges = None
    if use_ai_vad:
        speech_ranges = detect_speech_by_ai(input_file)

    if not speech_ranges:
        speech_ranges = detect_speech_by_db(input_file, total_dur, noise_db=noise_db)

    valid_ranges = []
    for st, et in speech_ranges:
        st = max(0.0, min(st, total_dur))
        et = max(0.0, min(et, total_dur))
        if et - st >= 0.18:
            valid_ranges.append((st, et))

    if not valid_ranges:
        print("⚠️ 未检测到有效人声！")
        return False

    kept_dur = sum(et - st for st, et in valid_ranges)
    removed_dur = total_dur - kept_dur

    print(f"\n📊 【中文扫描诊断报告】")
    print(f"  - 识别出有效讲话片段 : {len(valid_ranges)} 段")
    print(f"  - 原始视频总时长     : {total_dur:.2f} 秒")
    print(f"  - 切除无声空白时长   : ✂️ {removed_dur:.2f} 秒 (精简了 {removed_dur/total_dur*100:.1f}%)")
    print(f"  - 预计输出成片时长   : ⏱️ {kept_dur:.2f} 秒")

    temp_dir = input_file.parent / "temp_clips_work"
    temp_dir.mkdir(exist_ok=True)
    concat_list_file = temp_dir / "concat_list.txt"
    vocal_filter = get_vocal_enhancement_filter()

    print(f"\n🚀 [步骤 1/2] 正在裁切并美化 {len(valid_ranges)} 个中文有声片段...")
    clip_paths = []
    for idx, (st, et) in enumerate(valid_ranges, 1):
        clip_file = temp_dir / f"seg_{idx:04d}.mp4"
        dur = et - st
        cmd_seg = [
            "ffmpeg", "-y", "-ss", f"{st:.3f}", "-t", f"{dur:.3f}",
            "-i", str(input_file.resolve()), "-af", vocal_filter,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "256k", str(clip_file.resolve())
        ]
        r = subprocess.run(cmd_seg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        if r.returncode == 0 and clip_file.exists():
            clip_paths.append(clip_file)
            print(f"\r  -> 已完成片段处理: {idx}/{len(valid_ranges)}", end="", flush=True)

    print()
    if not clip_paths:
        return False

    lines = [f"file '{p.resolve().as_posix()}'\n" for p in clip_paths]
    concat_list_file.write_text("".join(lines), encoding="utf-8")

    print(f"🚀 [步骤 2/2] 正在无缝拼接所有精简片段...")
    cmd_merge = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file.resolve()), "-c", "copy",
        str(output_file.resolve())
    ]
    res = subprocess.run(cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

    for p in clip_paths:
        if p.exists():
            p.unlink()
    if concat_list_file.exists():
        concat_list_file.unlink()
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    if res.returncode == 0:
        print(f"\n🎉 成功！已输出精简中文成片：👉 {output_file.name}\n")
        return True
    return False


def interactive_menu():
    input_dir = os.environ.get("VIDEO_INPUT_DIR", str(Path.cwd()))
    while True:
        videos = scan_target_videos(input_dir)
        print("\n" + "=" * 65)
        print("      ✂️ 中文自媒体教程视频：AI 智能跳切 + 人声美化工具")
        print("=" * 65)
        print(f" 扫描目录 : {input_dir} (共发现 {len(videos)} 个视频)")
        print("-" * 65)
        print(" 1. 🤖 [AI 智能跳切] 神经网络识别中文人声，精准切除空白 [推荐]")
        print(" 2. 🔥 [强力分贝切除] 纯分贝强力切除停顿与呼吸气口")
        print(" 0. 退出程序")
        print("=" * 65)

        choice = input("请输入选项 (0-2): ").strip()
        if choice == "1" and videos:
            print("\n📂 请选择视频：")
            for idx, v in enumerate(videos, 1):
                print(f"  [{idx}] -> {v.name}")
            v_choice = input("请输入编号: ").strip()
            if v_choice.isdigit() and 1 <= int(v_choice) <= len(videos):
                target = videos[int(v_choice) - 1]
                process_tighten_video(target, target.parent / f"tightened_{target.name}", use_ai_vad=True)
                break
        elif choice == "2" and videos:
            print("\n📂 请选择视频：")
            for idx, v in enumerate(videos, 1):
                print(f"  [{idx}] -> {v.name}")
            v_choice = input("请输入编号: ").strip()
            if v_choice.isdigit() and 1 <= int(v_choice) <= len(videos):
                target = videos[int(v_choice) - 1]
                process_tighten_video(target, target.parent / f"tightened_strong_{target.name}", use_ai_vad=False)
                break
        elif choice == "0":
            sys.exit(0)


if __name__ == "__main__":
    interactive_menu()
