"""English-first UI copy. Language never participates in inference or cache identity."""
import html


COPY = {
    "language": ("Interface language", "界面语言"),
    "upload": ("Upload audio or video", "上传音频或视频"),
    "sample_hint": ("Leave empty to use the **30-second sample**.", "留空使用 **30 秒示例视频**。"),
    "sample_button": ("↺ Use sample video", "↺ 使用示例视频"),
    "model": ("Model", "模型"),
    "window": ("Window length / s", "窗口时长 / 秒"),
    "hop": ("Hop size / s", "窗口间隔 / 秒"),
    "settings_hint": ("Window: audio processed at a time. Hop: how far each window advances.", "窗口决定每次处理的时长；间隔决定窗口前进的步长。"),
    "reset": ("Default · 3 / 3 s", "默认 · 3 / 3 秒"),
    "overlap": ("Overlap · 3 / 1.5 s", "重叠 · 3 / 1.5 秒"),
    "run": ("Separate & visualize →", "分离并可视化 →"),
    "usage": ("Switch stems · Seek through the audio or video · Select a window\n\nThe chart shows the model’s actual frequency boundaries.", "切换音轨 · 拖动进度 · 点击窗口\n\n频带图显示模型的真实频率边界。"),
    "about": ("About the models", "关于模型"),
    "about_copy": ("**BSRNN / TIGER**\n\nSeparate speech, music and sound effects. Choose a model above; run again after changing the window or hop.", "**BSRNN / TIGER**\n\n分离人声、音乐与音效。可在上方切换模型；调整窗口或间隔后请重新运行。"),
    "finite": ("Enter a finite number of seconds.", "请输入有限的秒数。"),
    "window_range": ("Window length must be between 0.5 and 15 seconds.", "推理窗口时长应在 0.5–15 秒之间。"),
    "hop_range": ("Hop size must be at least 0.1 seconds and no longer than the window, so no audio is skipped.", "窗口间隔应至少为 0.1 秒，且不能大于窗口时长，以免漏掉音频。"),
    "choose_model": ("Select BSRNN or TIGER.", "请选择 BSRNN 或 TIGER。"),
    "missing_video": ("Upload audio or video, or add the sample video to the project folder.", "请上传音频或视频，或将示例视频放在项目目录中。"),
    "missing_weight": ("{model} weights were not found. Follow the README to finish setup.", "未找到 {model} 模型权重，请按 README 完成配置。"),
    "checking": ("Checking media and model", "核对媒体与模型"),
    "preparing": ("Preparing video and audio", "准备视频与音频"),
    "loading": ("Loading {model}", "加载 {model}"),
    "separating": ("Separating audio", "分离音轨"),
    "separating_window": ("Separating window {current}/{total}", "正在分离窗口 {current}/{total}"),
    "saving": ("Saving stems and frequency boundaries", "保存音轨与频带"),
    "packaging": ("Preparing downloads", "整理下载文件"),
    "complete": ("Complete", "完成"),
    "failed": ("Processing failed. Check the file and settings, then try again. Details are in the local server console.", "处理失败。请检查文件和推理设置后重试，详细原因可查看本地服务终端。"),
    "no_audio": ("This file has no audio track to separate.", "这个文件没有音轨，无法运行音源分离。"),
    "no_video": ("This file has no usable video track. Choose a video file.", "这个文件没有可用的视频轨道。请选择视频文件。"),
    "ffmpeg": ("FFmpeg or FFprobe was not found. Install FFmpeg and add it to PATH.", "找不到 FFmpeg 或 FFprobe。请安装 FFmpeg，并将其加入 PATH。"),
    "too_long": ("This file is too long. Choose a clip up to 600 seconds.", "媒体过长，请选择不超过 600 秒的片段。"),
}


def t(key, language="en", **values):
    return COPY[key][1 if language == "zh" else 0].format(**values)


def bi(en, zh):
    """A live-translatable HTML leaf; values and default text are escaped."""
    return (f'<span data-en="{html.escape(en, quote=True)}" '
            f'data-zh="{html.escape(zh, quote=True)}">{html.escape(en)}</span>')


def friendly_error(exc, language="en"):
    message = str(exc)
    # Keep expected validation errors specific; never surface checkpoint paths.
    for key, pair in COPY.items():
        if message in pair:
            return t(key, language)
    for model in ("BSRNN", "TIGER"):
        if message in (t("missing_weight", "en", model=model), t("missing_weight", "zh", model=model)):
            return t("missing_weight", language, model=model)
    if "没有音轨" in message or "没有可用的音轨" in message:
        return t("no_audio", language)
    if "没有可用的视频轨道" in message:
        return t("no_video", language)
    if "找不到" in message and ("ffmpeg" in message.lower() or "ffprobe" in message.lower()):
        return t("ffmpeg", language)
    if "最多支持" in message or ("超过" in message and "600" in message):
        return t("too_long", language)
    return t("failed", language)
