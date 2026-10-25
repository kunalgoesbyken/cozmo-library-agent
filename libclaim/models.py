from pydantic import BaseModel


class Still(BaseModel):
    seq: int
    t_ms: int
    path: str
    w: int
    h: int
    blur: float
    glare: float
    motion_dps: float
    pose: list[float] | None = None
    intrinsics: dict[str, float] | None = None
    depth_path: str | None = None
