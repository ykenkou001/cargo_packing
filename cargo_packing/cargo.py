"""
荷物を表すCargoクラスの定義
"""

from typing import Tuple


class Cargo:
    """荷物を表すクラス"""

    def __init__(
        self,
        id: int,
        length: float,
        width: float = None,
        height: float = None,
        weight: float = 0,
        priority: int = 0,
        rotatable: bool = True,
    ):
        self.id = id
        self.length = length  # 長さ/サイズ (x方向)
        self.width = width if width is not None else length  # 幅 (y方向)
        self.height = height if height is not None else length  # 高さ (z方向)
        self.weight = weight  # 重量
        self.priority = priority  # 優先度（大きいほど優先）
        self.position = None  # 配置位置 (x, y, z)
        self.rotation = (0, 0, 0)  # 回転状態 (x, y, z軸周りの回転)
        self.rotatable = rotatable  # 回転可能かどうか
        self.original_dimensions = (length, width, height)  # 元のサイズを保存

    def get_dimensions(self) -> Tuple[float, float, float]:
        """現在の回転状態での寸法を取得"""
        if self.rotation == (0, 0, 0):
            return self.original_dimensions

        # 回転に応じた寸法を返す
        length, width, height = self.original_dimensions
        rx, ry, rz = self.rotation

        # 単純化のため、90度単位の回転のみサポート
        if rz % 2 == 1:  # z軸周りに90度/270度回転
            length, width = width, length

        if rx % 2 == 1:  # x軸周りに90度/270度回転
            width, height = height, width

        if ry % 2 == 1:  # y軸周りに90度/270度回転
            length, height = height, length

        return length, width, height

    def volume(self) -> float:
        """荷物の体積を計算"""
        return self.length * self.width * self.height

    def rotate(self, rx: int = 0, ry: int = 0, rz: int = 0) -> bool:
        """荷物を回転させる (0=0度, 1=90度, 2=180度, 3=270度)"""
        if not self.rotatable:
            return False

        # 回転制約をチェック（例：特定の面を上にする必要がある場合など）
        # 簡略化のため、ここでは制約なしとする

        self.rotation = (
            (self.rotation[0] + rx) % 4,
            (self.rotation[1] + ry) % 4,
            (self.rotation[2] + rz) % 4,
        )

        # 回転に応じて寸法を更新
        length, width, height = self.get_dimensions()
        self.length, self.width, self.height = length, width, height

        return True

    def __repr__(self):
        return f"Cargo(id={self.id}, dims=({self.length}, {self.width}, {self.height}), weight={self.weight}, rot={self.rotation})"
