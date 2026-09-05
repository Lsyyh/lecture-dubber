# Windows bootstrap equivalent of bootstrap_ubuntu.sh.
# Run from the repository root with the target conda env active, e.g.:
#   conda activate itksnap-dls
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows.ps1
#
# Notes vs Linux:
# - ffmpeg comes from the vendored gyan.dev build in tools\ffmpeg (no admin needed, libass included).
# - No new conda/venv is created; the script installs into the ACTIVE environment,
#   reusing its existing CUDA PyTorch build (torch/torchaudio are NOT upgraded).
# - All weights land under pretrained_models\ and third_party\ on D:, never C:.
# - Downloads use mirrors first: PyPI -> mirrors.huaweicloud.com (pip global config),
#   HuggingFace -> hf-mirror.com, GitHub -> ghfast.top, models -> ModelScope.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path

# 1. ffmpeg: download + extract into tools\ffmpeg if missing
if (-not (Test-Path "$Root\tools\ffmpeg\bin\ffmpeg.exe")) {
    New-Item -ItemType Directory -Force -Path "$Root\tools" | Out-Null
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" `
        -OutFile "$Root\tools\ffmpeg-essentials.zip"
    Expand-Archive "$Root\tools\ffmpeg-essentials.zip" "$Root\tools\ffmpeg-tmp" -Force
    Move-Item "$Root\tools\ffmpeg-tmp\*" "$Root\tools\ffmpeg" -Force
    Remove-Item "$Root\tools\ffmpeg-tmp" -Recurse -Force -ErrorAction SilentlyContinue
}

# 2. Python deps: base package + WhisperX (pinned to keep the env's existing torch)
#    plus the minimal CosyVoice runtime imports. Do NOT install third_party\CosyVoice\requirements.txt:
#    it pins torch==2.3.1 / numpy==1.26.4 and would break this environment.
pip install -e $Root
pip install "whisperx==3.3.4" "pyannote.audio<4" "transformers<5" "huggingface_hub<1" "numpy==2.3.5" `
    "nvidia-cudnn-cu12==8.9.7.29" nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 `
    modelscope conformer diffusers inflect wetext openai-whisper librosa einops soundfile

# 3. Config
if (-not (Test-Path "$Root\config.yaml")) {
    Copy-Item "$Root\config.example.yaml" "$Root\config.yaml"
}

Write-Host "Base environment ready."
Write-Host "Next: scripts\get_models.ps1 (weights via ModelScope/hf-mirror), scripts\start_llm_server.cmd, scripts\dubber.cmd run <video|url>"
