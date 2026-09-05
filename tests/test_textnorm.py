from lecture_dubber.textnorm import spoken_form


def test_greek_letters_are_spoken():
    assert spoken_form("关于θ的函数") == "关于西塔的函数"
    assert spoken_form("α和β") == "阿尔法和贝塔"
    assert spoken_form("Δy的变化") == "德尔塔y的变化"
    assert spoken_form("ω是角频率") == "欧米伽是角频率"


def test_math_symbols_are_spoken():
    assert spoken_form("x²") == "x的平方"
    assert spoken_form("3×4") == "3乘4"
    assert spoken_form("45°") == "45度"
    assert spoken_form("λ≈1") == "拉姆达约等于1"


def test_unit_symbols_beat_single_letters():
    assert spoken_form("波长1μm") == "波长1微米"
    assert spoken_form("μs级延迟") == "微秒级延迟"


def test_latex_names_are_spoken():
    assert spoken_form(r"\theta与\alpha") == "西塔与阿尔法"
    assert spoken_form(r"\Delta L") == "德尔塔 L"


def test_plain_text_unchanged():
    assert spoken_form("梯度下降是一种迭代优化算法。") == "梯度下降是一种迭代优化算法。"
