@echo off
rem Windows launcher for the dubber CLI.
rem Usage: scripts\dubber.cmd <dubber arguments, e.g. run video.mp4 -o outputs/job>
setlocal
set "ROOT=%~dp0.."
set "PYENV=D:\miniconda\envs\itksnap-dls"
if defined DUBBER_PYENV set "PYENV=%DUBBER_PYENV%"

rem ffmpeg/ffprobe from the vendored build
set "PATH=%ROOT%\tools\ffmpeg\bin;%PATH%"

rem cuDNN/cuBLAS/cuDRT DLLs required by ctranslate2 (faster-whisper)
set "NVIDIA_BIN=%PYENV%\Lib\site-packages\nvidia"
set "PATH=%NVIDIA_BIN%\cudnn\bin;%NVIDIA_BIN%\cublas\bin;%NVIDIA_BIN%\cuda_runtime\bin;%PATH%"

rem HuggingFace downloads: mirror endpoint + cache on D: (never C:)
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%ROOT%\pretrained_models\hf-home"
set "HF_HUB_ENABLE_HF_TRANSFER=0"

rem CosyVoice repo must be found from any working directory
pushd "%ROOT%"
"%PYENV%\Scripts\dubber.exe" %*
popd
endlocal
