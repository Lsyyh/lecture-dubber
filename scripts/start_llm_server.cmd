@echo off
rem Start llama-server (OpenAI-compatible) with the local Qwen3 GGUF on port 8000.
rem Keep this running while `dubber run` translates.
rem Qwen3-8B is a hybrid-thinking model; enable_thinking=false keeps translation fast.
setlocal
set "ROOT=%~dp0.."
set "MODEL_DIR=%ROOT%\pretrained_models\gguf"
set "MODEL="
for %%f in ("%MODEL_DIR%\*.gguf") do (
    if not defined MODEL set "MODEL=%%~ff"
)
if not defined MODEL (
    echo No GGUF model found under %MODEL_DIR%
    exit /b 1
)
echo Serving %MODEL% on http://127.0.0.1:8000/v1
"%ROOT%\tools\llama-cpp\llama-server.exe" ^
  --host 127.0.0.1 --port 8000 ^
  -m "%MODEL%" -ngl 99 -c 8192 --threads 8 ^
  --chat-template-kwargs "{\"enable_thinking\":false}"
endlocal
