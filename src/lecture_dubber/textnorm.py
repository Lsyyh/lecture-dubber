"""Map Greek letters and math symbols to spoken Mandarin for TTS.

The translation keeps symbols like theta as-is for the subtitles, so the spoken
form must be produced right before synthesis. Subtitle text is never affected
because this runs only inside the TTS adapter.
"""
from __future__ import annotations

import re

_GREEK: dict[str, str] = {
    "α": "阿尔法", "β": "贝塔", "γ": "伽马", "δ": "德尔塔", "ε": "艾普西隆",
    "ζ": "泽塔", "η": "伊塔", "θ": "西塔", "ι": "约塔", "κ": "卡帕",
    "λ": "拉姆达", "μ": "缪", "ν": "纽", "ξ": "克西", "ο": "奥米克戎",
    "π": "派", "ρ": "柔", "σ": "西格玛", "τ": "陶", "υ": "宇普西隆",
    "φ": "斐", "χ": "凯", "ψ": "普赛", "ω": "欧米伽",
    "Γ": "伽马", "Δ": "德尔塔", "Θ": "西塔", "Λ": "拉姆达", "Ξ": "克西",
    "Π": "派", "Σ": "西格玛", "Φ": "斐", "Ψ": "普赛", "Ω": "欧米伽",
}

# Longer keys must win over single letters, e.g. "μm" before "μ".
_SYMBOLS: dict[str, str] = {
    "μm": "微米", "μs": "微秒", "μg": "微克", "℃": "摄氏度",
    "∂": "偏", "∑": "求和", "∏": "连乘", "∫": "积分", "√": "根号",
    "∞": "无穷大", "≈": "约等于", "≠": "不等于", "≤": "小于等于", "≥": "大于等于",
    "±": "正负", "×": "乘", "÷": "除以", "°": "度", "∈": "属于",
    "²": "的平方", "³": "的立方",
}

_LATEX_GREEK: dict[str, str] = {
    "alpha": "阿尔法", "beta": "贝塔", "gamma": "伽马", "delta": "德尔塔",
    "epsilon": "艾普西隆", "varepsilon": "艾普西隆", "zeta": "泽塔", "eta": "伊塔",
    "theta": "西塔", "iota": "约塔", "kappa": "卡帕", "lambda": "拉姆达",
    "mu": "缪", "nu": "纽", "xi": "克西", "omicron": "奥米克戎", "pi": "派",
    "rho": "柔", "sigma": "西格玛", "tau": "陶", "upsilon": "宇普西隆",
    "phi": "斐", "varphi": "斐", "chi": "凯", "psi": "普赛", "omega": "欧米伽",
    "Gamma": "伽马", "Delta": "德尔塔", "Theta": "西塔", "Lambda": "拉姆达",
    "Xi": "克西", "Pi": "派", "Sigma": "西格玛", "Upsilon": "宇普西隆",
    "Phi": "斐", "Psi": "普赛", "Omega": "欧米伽",
}

_SYMBOL_RE = re.compile(
    "|".join(sorted(map(re.escape, list(_GREEK) + list(_SYMBOLS)), key=len, reverse=True))
)
_LATEX_RE = re.compile(r"\\([A-Za-z]+)")


def spoken_form(text: str) -> str:
    """Return text with Greek letters and math symbols expanded to spoken form."""
    text = _LATEX_RE.sub(lambda m: _LATEX_GREEK.get(m.group(1), m.group(0)), text)
    return _SYMBOL_RE.sub(
        lambda m: _GREEK.get(m.group(0)) or _SYMBOLS[m.group(0)], text
    )
