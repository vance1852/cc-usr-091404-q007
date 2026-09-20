"""api 与服务的公共异常类型。"""


class PotencyError(Exception):
    """系统内所有业务异常的基类。"""


class NotFoundError(PotencyError):
    """实体不存在。"""


class DuplicatePlateError(PotencyError):
    """检测到重复导入的板（内容哈希一致）。"""

    def __init__(self, message: str, existing_plate_id: int):
        super().__init__(message)
        self.existing_plate_id = existing_plate_id


class WorkflowError(PotencyError):
    """违反工作流约束（状态、角色、顺序等）。"""


class ValidationError(PotencyError):
    """输入数据校验失败。"""
