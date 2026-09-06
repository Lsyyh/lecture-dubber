# Download all model weights into the repository (D: drive).
# All models come from ModelScope mirrors (direct HTTP from huggingface.co and even
# hf-mirror.com is unreliable on this network; ModelScope API is fast and stable).
# Final deployment set: faster-whisper-large-v3 + wav2vec2 aligner (ASR),
# Qwen3-VL-2B-Instruct Q8 GGUF (translate + subtitle QC), CosyVoice-300M-SFT (TTS).
# Requires: conda env active (modelscope installed).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path
New-Item -ItemType Directory -Force -Path "$Root\pretrained_models" | Out-Null
$ms = "modelscope"

# 1. WhisperX ASR: faster-whisper large-v3 (~3.1 GB; community mirror of Systran/faster-whisper-large-v3)
& $ms download --model gpustack/faster-whisper-large-v3 `
    --include "config.json" "model.bin" "preprocessor_config.json" "tokenizer.json" "vocabulary.json" `
    --local_dir "$Root\pretrained_models\faster-whisper-large-v3"

# 2. WhisperX English alignment model (~1.3 GB; mirror of jonatasgrosman/wav2vec2-large-xlsr-53-english)
& $ms download --model AI-ModelScope/wav2vec2-large-xlsr-53-english `
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json" "alphabet.json" "vocab.json" "special_tokens_map.json" `
    --local_dir "$Root\pretrained_models\wav2vec2-large-xlsr-53-english"

# 3. Qwen3-VL-2B-Instruct Q8 GGUF + mmproj (~3.3 GB) for llama-server (translate + VLM QC)
& $ms download --model Qwen/Qwen3-VL-2B-Instruct-GGUF `
    --include "Qwen3VL-2B-Instruct-Q8_0.gguf" "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf" `
    --local_dir "$Root\pretrained_models\gguf"

# 4. CosyVoice-300M-SFT TTS (~5.4 GB, fastest backend measured; see AGENTS.md)
& $ms download --model iic/CosyVoice-300M-SFT --local_dir "$Root\pretrained_models\CosyVoice-300M-SFT"

Write-Host "All weights ready under $Root\pretrained_models"
