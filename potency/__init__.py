"""效价检测会审系统。

- potency.fitting   确定性 4PL 曲线拟合（Levenberg-Marquardt）
- potency.metrics   平行性、精密度、有效范围、相对效价
- potency.rules     版本化规则集（无效板判定）
- potency.combine   批次结论组合策略
- potency.services  业务工作流（导入、分析、排除、复测、结论、追溯）
- potency.api       FastAPI 接口
"""
from .services import PotencyService

__all__ = ["PotencyService"]
__version__ = "0.1.0"
