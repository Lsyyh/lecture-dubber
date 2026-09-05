# Download all model weights into the repository (D: drive).
# All models come from ModelScope mirrors (direct HTTP from huggingface.co and even
# hf-mirror.com is unreliable on this network; ModelScope API is fast and stable).
# Requires: conda env active (modelscope installed).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path
New-Item -ItemType Directory -Force -Path "$Root\pretrained_models" | Out-Null
$ms = "modelscope"

# 1. CosyVoice-300M-SFT TTS model (~4.8 GB)
& $ms download --model iic/CosyVoice-300M-SFT --local_dir "$Root\pretrained_models\CosyVoice-300M-SFT"

# 2. faster-whisper large-v3 ASR model (~3.1 GB; community mirror of Systran/faster-whisper-large-v3)
& $ms download --model gpustack/faster-whisper-large-v3 `
    --include "config.json" "model.bin" "preprocessor_config.json" "tokenizer.json" "vocabulary.json" `
    --local_dir "$Root\pretrained_models\faster-whisper-large-v3"

# 3. WhisperX English alignment model (~1.3 GB; mirror of jonatasgrosman/wav2vec2-large-xlsr-53-english)
& $ms download --model AI-ModelScope/wav2vec2-large-xlsr-53-english `
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json" "alphabet.json" "vocab.json" "special_tokens_map.json" `
    --local_dir "$Root\pretrained_models\wav2vec2-large-xlsr-53-english"

# 4. Qwen3-8B GGUF for llama-server (~6.6 GB; Q6_K single file)
& $ms download --model Qwen/Qwen3-8B-GGUF --include "Qwen3-8B-Q6_K.gguf" `
    --local_dir "$Root\pretrained_models\gguf"

Write-Host "All weights ready under $Root\pretrained_models"
