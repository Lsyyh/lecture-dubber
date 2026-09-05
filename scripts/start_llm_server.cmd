@echo off
rem Start llama-server (OpenAI-compatible) with the local Qwen3-VL GGUF on port 8000.
rem Serves both translation and VLM subtitle QC. Keep running while `dubber run` works.
rem Qwen3-VL is a hybrid-thinking model; enable_thinking=false keeps translation fast.
setlocal
set "ROOT=%~dp0.."
set "MODEL_DIR=%ROOT%\pretrained_models\gguf"
set "MODEL="
for %%f in ("%MODEL_DIR%\Qwen*.gguf") do (
    if not defined MODEL set "MODEL=%%~ff"
)
if not defined MODEL (
    echo No GGUF model found under %MODEL_DIR%
    exit /b 1
)
set "MMPROJ="
for %%f in ("%MODEL_DIR%\mmproj-*.gguf") do set "MMPROJ=%%~ff"
set "MMARGS="
if defined MMPROJ set "MMARGS=--mmproj %MMPROJ%"
echo Serving %MODEL% on http://127.0.0.1:8000/v1 (mmproj: %MMPROJ%)
"%ROOT%\tools\llama-cpp\llama-server.exe" ^
  --host 127.0.0.1 --port 8000 ^
  -m "%MODEL%" %MMARGS% -ngl 99 -c 16384 -np 4 --threads 8 ^
  --chat-template-kwargs "{\"enable_thinking\":false}"
endlocal
