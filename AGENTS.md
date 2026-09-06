# AGENTS.md

本文件是 ZCode 工作区指令，只记录本项目特有的事实与规则；个人通用习惯见 `~/.zcode/AGENTS.md`。编码习惯参照 `D:\Code\harness-pilot\CLAUDE.md`（见文末）。

## 项目概述

- **lecture-dubber**：本地优先的讲座视频跨语言配音流水线（英→中）。链路：yt-dlp/本地导入 → ffmpeg 提音频 → WhisperX ASR + 强制对齐 → 4–15s 语义分段合并 → Qwen（OpenAI 兼容 HTTP）翻译 → CosyVoice 中文 TTS → 时长拟合 → 时间线合成 + 中文 SRT → ffmpeg 渲染 MP4。
- 刻意不用 agent 框架、数据库、队列，不做口型同步；每个昂贵阶段的产物都落盘缓存，`dubber run` 默认断点续跑。
- src layout：源码在 `src/lecture_dubber/`（扁平模块，一个模块一个职责），打包用 hatchling，Python >= 3.10。

## 常用命令

```bash
pip install -e '.[dev]'           # 基础安装（typer/pydantic/httpx/yt-dlp 等）
pip install -e '.[asr]'           # 追加 WhisperX/torch（仅 ASR 阶段需要）
make test                         # PYTHONPATH=src pytest -q；单文件: pytest tests/test_segment.py -q
make doctor                       # CLI import 冒烟；运行环境体检用 dubber doctor
ruff check --fix && ruff format   # line-length=100 已在 pyproject 配置
dubber run <本地视频|URL> [-o job_dir] [--no-resume] [-c config.yaml]
dubber fetch <URL> -o <dir>       # 只下载不处理
```

测试只覆盖纯逻辑（分段合并、字幕），不依赖 whisperx/torch/CosyVoice，任何机器可跑。

## 架构边界

- `cli.py`（typer：run/fetch/doctor）→ `pipeline.py::Pipeline` 六阶段编排 → 各功能模块。全部数据结构与配置（`Segment` / `TranslationUnit` / `SourceInfo` / `JobState` / `Config`）集中在 `models.py`。
- **resume 语义不可破坏**：job 目录下 `state.json`、`audio.wav`、`segments.json`、`units.json`、`tts/`、`tts_fit/`、`zh.srt`、`dub.wav`、`final.zh.mp4` 是逐阶段缓存；每阶段先查缓存再计算，产物写盘后立即回写 `state.json` / `units.json`。改流水线时保持"先查缓存、写盘即落 state"的模式。
- **URL 输入的默认 job 目录是 `outputs/url-job`（所有 URL 共用）**，跑不同 URL 必须用 `-o` 指定目录，否则缓存互相覆盖。
- `models.py::Config` 拥有全部配置字段与默认值；`config.yaml` 只是覆盖层（`config.py` 做 YAML → pydantic 校验，文件缺失时用纯默认值）。新增配置项：`Config` 加字段 + `config.example.yaml` 同步默认值。
- **重依赖一律函数内懒导入**：`asr.py`（whisperx/torch）、`tts.py`（cosyvoice）、`source.py`（yt_dlp）。基础安装不装它们也能 import 和跑测试；新加重依赖沿用此模式，不要提到模块顶层。
- 所有 ffmpeg/ffprobe 调用只在 `media.py`；外部命令用 `utils.ensure_command` 检查；字幕渲染滤镜需要带 libass 的 ffmpeg。
- `tts.py` 的 `sys.path.insert`（指向 `third_party/CosyVoice` 及其 Matcha-TTS）是唯一刻意的 path hack——外接仓库适配器；CosyVoice 上游 API 变化时只改该文件。
- 翻译-时长闭环（pipeline 第 5 阶段）：合成 → 实测/目标时长比 > `duration_rewrite_threshold`(1.22) 且 attempts 未用尽 → LLM 压缩重译再合成；否则 `fit_audio` 用 atempo 拉伸（超出 soft 区间则原样拷贝）。`units.json` 是人工改译检查点：改完译文删对应 `tts_fit/NNNNN.wav` 再重跑。
- `glossary.yaml` 整体进翻译 prompt；处理新课前先编辑术语表，对成片质量影响最大。

## Windows 本机部署（已完成，2026-09）

完整流水线已在本机 Windows + RTX 3090 跑通，组件与入口：

- **Python 环境**：复用 conda 环境 `D:\miniconda\envs\itksnap-dls`（py3.12 + torch 2.9.1+cu130，**torch/torchaudio/numpy 保持原版本未被升级**）。whisperx 固定 3.3.4 + pyannote<4 + transformers<5，否则依赖链会降级 torch。
- **启动入口**：一律走 `scripts\dubber.cmd`（设置 vendored ffmpeg PATH、ctranslate2 所需的 nvidia DLL PATH、`HF_ENDPOINT=hf-mirror.com`、`HF_HOME` 指到 `pretrained_models\hf-home`，并 cd 到仓库根）。LLM 服务用 `scripts\start_llm_server.cmd`。权重下载脚本 `scripts\get_models.ps1`，Windows 引导说明 `scripts\bootstrap_windows.ps1`。
- **权重全部在 `pretrained_models\`**（gitignore）：`faster-whisper-large-v3`、`wav2vec2-large-xlsr-53-english`（本地目录，`config.yaml` 的 `asr_model`/`asr_align_model` 直接指向它们）、`CosyVoice-300M-SFT`、`gguf\Qwen3-8B-Q6_K.gguf`。`third_party\` 下是 CosyVoice + Matcha-TTS 克隆，`tools\` 下是 ffmpeg 与 llama-cpp。
- **镜像路线**：huggingface.co 与 github.com 直连不通，且 huggingface_hub 对 hf-mirror.com 的 HEAD 元数据校验会失败——**模型一律走 ModelScope**（iic、gpustack、AI-ModelScope、Qwen 官方），代码走 ghfast.top / ghproxy.net / gh-proxy.com（大文件用分段下载），pip 走华为云镜像。
- **LLM/VLM 服务**：`tools\llama-cpp` 的 llama-server（CUDA 构建）+ **Qwen3-VL-2B-Instruct Q8**（ModelScope，含 mmproj），一个服务同时承担翻译和字幕质控，OpenAI 兼容接口 127.0.0.1:8000/v1；启动参数带 `-np 1`（**不要开并发槽**：b10795 的并发响应会串话）和 `--chat-template-kwargs {"enable_thinking":false}`。**`response_format` 的 `json_object` 在此构建上不生效**（模型自由输出会跑偏），**必须用 `json_schema`**（服务端 grammar 强制，已验证）；请求体要 `ensure_ascii=False` 手工序列化（httpx 的 `json=` 会把中文转成 \uXXXX，小模型读不懂）。多模态输出可能仍带 ``` 围栏——解析一律走 `utils.extract_json`。
- **Web UI**（`webui.py` + `dubber serve` / `scripts\start_webui.cmd`，端口 8010）：单页前端提交任务（本地文件/B 站缓存目录/URL）、实时进度（从 job 目录缓存文件派生，与 CLI 共享状态）、**单元级边处理边播放**（tts_fit wav 就绪即可点播）、成片 Range 流播放、任务取消（`Pipeline(stop_event=...)`）。单活动任务；进度函数 `job_status` 有单测。
- **VLM 质控**（`qc.py`，`subtitle_alignment: auto` 时启用）：渲染前采样帧检测源视频有无烧录硬字幕（决定我们字幕放顶/底，后台线程与翻译阶段并行）；渲染后检查中文字幕是否完整、有无乱码遮挡，结果写 `qc_report.json`。QC 任何失败都不能中断流水线。
- **TTS 定案**（2026-09 实测）：**CosyVoice1-300M-SFT（50Hz）最快，avg 4.9s/句，保留为唯一后端**。CosyVoice2-0.5B 实测慢 4 倍（24s/句，无 TRT 时 flow matching 太重）已删；CosyVoice-300M-25Hz zero-shot 实测 8.3s/句（零样本每句重新提取 prompt 特征）已删。`tts.py` 保留零样本双路径（配置 `cosyvoice_prompt_wav` 即克隆音色，参考音频 `pretrained_models/prompt_wav/zh_female.wav` 由 SFT 中文女生成）；CosyVoice 官方代码对已加载 tensor 重复调用 `load_wav`，tts.py 打了 tensor 直通补丁。若未来要换 TTS：F5-TTS 依赖树会强升 transformers 5.x 破坏 whisperx，勿装。
- **流水线并行**：翻译线程池（`translate_workers`）；翻译→TTS 重叠（as_completed 到一个译完一个就合成）；硬字幕检测后台线程；`render_preset` 可配（veryfast 省一半编码时间）；`cleanup_intermediates: true` 成功后自动删 `tts/`、`dub.wav`、`source/*.m4s`。
- **兼容垫片**：`src/lecture_dubber/_torchaudio_compat.py` 在导入 whisperx 前补回 torchaudio 2.9 删除的 `info/load/AudioMetaData`（soundfile 实现，供 pyannote VAD 使用），并让 lightning 以 `weights_only=False` 加载 whisperx 内置 VAD 检查点。`tts.py` 用 soundfile 写 wav（torchaudio.save 在 2.9 需要 torchcodec）。CosyVoice 上游若更新，重查这两处。
- **ffmpeg**：vendored `tools\ffmpeg\bin`（gyan essentials 9.0.1，含 libass），任何调用方都必须把它放 PATH 最前，否则 whisperx 的 load_audio 会在中文 Windows 上因 GBK stderr 解码崩溃。ffmpeg 9 已移除 `-filter_complex_script`，多路 filtergraph 要用 `-/filter_complex <file>` 新语法（`media.compose_timeline` 已如此）；77 路输入即使这样也要用相对路径 + cwd，否则命令行超 8191 字符被 cmd 劫持路径拒绝。
- **本机 CreateProcess 被劫持重走 cmd.exe**（python→python 传 `|` 参数可复现，根因未查明，疑似安全软件/系统策略）：未加引号的 `|`、`;`、`&` 会被 cmd 当作管道/分隔符拆开。因此 `utils.run` 自行构建命令行并对含元字符的参数按 MSVCRT 规则加引号，并对子进程输出 `errors="replace"`（ffmpeg 报错文本是 GBK）。新增 subprocess 调用必须走 `utils.run`，不要直接用 `subprocess`。

## 平台与环境坑

- 本工作区是 Windows 开发机，`dubber run` 全流程可直接在本机跑；Linux 部署参考 README 与 `scripts/bootstrap_ubuntu.sh`。
- 显存管理：WhisperX 主模型与对齐模型用完立即 `del` + `gc.collect()` + `torch.cuda.empty_cache()`，新增模型加载沿用这个模式。
- config 中的 endpoint / model / 说话人名（`中文女`）原样使用，不擅自替换。
- `outputs/`、`third_party/`、`pretrained_models/`、`tools/`、`.venv/` 已 gitignore；下载的媒体与模型权重不入库。

## 编码习惯（源自 harness-pilot CLAUDE.md，叠加在全局规则之上）

- 效率优先：不做防御性过度设计，只写防止静默错误的必要测试与边界检查。
- 面向公众的产物（注释、docstring、README、**commit message**）全英文；对话与内部工作文档用中文。本条 commit message 规则覆盖全局的中文 commit 规则；每个有效改动后本地 commit 一次。
- 复杂参数传递用参数类（本仓库用 pydantic BaseModel，如 `Config`），不用长 kwargs 链。
- 运行配置走 YAML（`config.yaml`）；README 保持精简（简介、快速开始、致谢），细节写内部文档。
- 删除、覆盖等不可逆操作先与用户确认。
