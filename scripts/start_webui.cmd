@echo off
rem Start the lecture-dubber web UI on http://127.0.0.1:8010
rem Requires start_llm_server.cmd to be running for translation/QC.
setlocal
set "ROOT=%~dp0.."
set "PYENV=D:\miniconda\envs\itksnap-dls"
if defined DUBBER_PYENV set "PYENV=%DUBBER_PYENV%"

set "PATH=%ROOT%\tools\ffmpeg\bin;%PATH%"
set "NVIDIA_BIN=%PYENV%\Lib\site-packages\nvidia"
set "PATH=%NVIDIA_BIN%\cudnn\bin;%NVIDIA_BIN%\cublas\bin;%NVIDIA_BIN%\cuda_runtime\bin;%PATH%"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%ROOT%\pretrained_models\hf-home"

pushd "%ROOT%"
echo Web UI on http://127.0.0.1:8010  (Ctrl+C to stop)
"%PYENV%\Scripts\dubber.exe" serve %*
popd
endlocal
