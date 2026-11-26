from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties

from conceptgraph.project_paths import PROJECT_ROOT


def configure_chinese_font() -> FontProperties:
    candidates = (
        PROJECT_ROOT / "checkpoint/fonts/NotoSansCJKsc-Regular.otf",
        PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct/SimSun.ttf",
        PROJECT_ROOT / "SimSun.ttf",
    )
    rcParams["axes.unicode_minus"] = False
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            font = FontProperties(fname=str(path))
            rcParams["font.family"] = [font.get_name()]
            return font
    return FontProperties()
