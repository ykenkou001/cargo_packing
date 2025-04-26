"""
コンテナを表すContainerクラスの定義
"""

import copy
from typing import Optional, Tuple

import numpy as np

from cargo_packing.cargo import Cargo


class Container:
    """コンテナを表すクラス"""

    def __init__(self, length: float, width: float, height: float, max_weight: float):
        self.length = length  # コンテナの長さ (x方向)
        self.width = width  # コンテナの幅 (y方向)
        self.height = height  # コンテナの高さ (z方向)
        self.max_weight = max_weight  # 最大積載重量
        self.cargo_list: list = []  # 積載された荷物リスト
        self.current_weight = 0  # 現在の積載重量
        # 3次元の空間を表す配列 (簡易的な実装のため)
        self.space_occupied = np.zeros(
            (int(length * 10) + 1, int(width * 10) + 1, int(height * 10) + 1),
            dtype=bool,
        )
        self.weight_distribution = np.zeros((int(length * 10) + 1, int(width * 10) + 1))
        # COGは重心（Center of Gravity）
        self.cog_x = 0.0  # 重心のx座標
        self.cog_y = 0.0  # 重心のy座標

    def volume(self) -> float:
        """コンテナの体積を計算"""
        return self.length * self.width * self.height

    def used_volume(self) -> float:
        """使用されている体積を計算"""
        total_volume = 0
        for cargo in self.cargo_list:
            total_volume += cargo.volume()
        return total_volume

    def space_utilization(self) -> float:
        """空間利用率を計算"""
        return self.used_volume() / self.volume() * 100

    def weight_utilization(self) -> float:
        """重量利用率を計算"""
        return self.current_weight / self.max_weight * 100

    def weight_balance_score(self) -> float:
        """重量バランスのスコアを計算（低いほど良い）"""
        if not self.cargo_list:
            return 0.0

        # 理想的な重心位置（コンテナの中心）
        ideal_cog_x = self.length / 2
        ideal_cog_y = self.width / 2

        # 現在の重心との差を正規化
        x_diff = abs(self.cog_x - ideal_cog_x) / self.length
        y_diff = abs(self.cog_y - ideal_cog_y) / self.width

        # 0〜1のスコア（0が最良、1が最悪）
        return (x_diff + y_diff) / 2

    def update_weight_distribution(self):
        """重量分布と重心を更新"""
        total_weight = 0
        weight_moment_x = 0
        weight_moment_y = 0

        # 重量分布をリセット
        self.weight_distribution = np.zeros(
            (int(self.length * 10) + 1, int(self.width * 10) + 1)
        )

        for cargo in self.cargo_list:
            if not cargo.position:
                continue

            x, y, _ = cargo.position
            x_center = x + cargo.length / 2
            y_center = y + cargo.width / 2

            # 重量モーメントを加算
            weight_moment_x += cargo.weight * x_center
            weight_moment_y += cargo.weight * y_center
            total_weight += cargo.weight

            # 重量分布を更新
            x1, y1 = int(x * 10), int(y * 10)
            x2, y2 = int((x + cargo.length) * 10), int((y + cargo.width) * 10)

            area = (x2 - x1) * (y2 - y1)
            if area > 0:
                weight_per_unit = cargo.weight / area
                self.weight_distribution[x1:x2, y1:y2] += weight_per_unit

        # 重心を計算
        if total_weight > 0:
            self.cog_x = weight_moment_x / total_weight
            self.cog_y = weight_moment_y / total_weight

    def is_position_valid(self, x: float, y: float, z: float, cargo: Cargo) -> bool:
        """指定位置に荷物を配置できるかチェック"""
        # コンテナ内に収まるかをチェック
        if (
            x + cargo.length > self.length
            or y + cargo.width > self.width
            or z + cargo.height > self.height
        ):
            return False

        # 簡易的な衝突検出（実際のアプリケーションではより精緻な実装が必要）
        x1, y1, z1 = int(x * 10), int(y * 10), int(z * 10)
        x2, y2, z2 = (
            int((x + cargo.length) * 10),
            int((y + cargo.width) * 10),
            int((z + cargo.height) * 10),
        )

        if np.any(self.space_occupied[x1:x2, y1:y2, z1:z2]):
            return False

        # 物理的な安定性のチェック（簡易版）
        if z > 0:  # 最下層でない場合、下に十分なサポートがあるか確認
            # 荷物の底面グリッド
            bottom = self.space_occupied[x1:x2, y1:y2, z1 - 1]
            support_area = np.sum(bottom)
            total_area = (x2 - x1) * (y2 - y1)

            # 底面の30%以上がサポートされていることを要求
            if support_area < total_area * 0.3:
                return False

        return True

    def can_load(self, cargo: Cargo) -> bool:
        """荷物が積載可能かチェック（重量のみ）"""
        return self.current_weight + cargo.weight <= self.max_weight

    def find_valid_position(self, cargo: Cargo) -> Optional[Tuple[float, float, float]]:
        """有効な配置位置を探索（最下層・最前列優先）"""
        for z in np.arange(0, self.height - cargo.height + 0.1, 0.1):
            for y in np.arange(0, self.width - cargo.width + 0.1, 0.1):
                for x in np.arange(0, self.length - cargo.length + 0.1, 0.1):
                    if self.is_position_valid(x, y, z, cargo):
                        return (x, y, z)
        return None

    def load_cargo(self, cargo: Cargo, position: Tuple[float, float, float]) -> bool:
        """荷物を指定位置に積載"""
        x, y, z = position

        if not self.can_load(cargo) or not self.is_position_valid(x, y, z, cargo):
            return False

        cargo.position = position
        self.cargo_list.append(cargo)
        self.current_weight += cargo.weight

        # 占有スペースを更新
        x1, y1, z1 = int(x * 10), int(y * 10), int(z * 10)
        x2, y2, z2 = (
            int((x + cargo.length) * 10),
            int((y + cargo.width) * 10),
            int((z + cargo.height) * 10),
        )
        self.space_occupied[x1:x2, y1:y2, z1:z2] = True

        # 重量分布を更新
        self.update_weight_distribution()

        return True

    def clone(self) -> "Container":
        """コンテナの複製を作成"""
        new_container = Container(self.length, self.width, self.height, self.max_weight)
        new_container.current_weight = self.current_weight
        new_container.space_occupied = np.copy(self.space_occupied)
        new_container.weight_distribution = np.copy(self.weight_distribution)
        new_container.cog_x = self.cog_x
        new_container.cog_y = self.cog_y

        # 荷物リストを複製
        for cargo in self.cargo_list:
            cargo_copy = copy.deepcopy(cargo)
            new_container.cargo_list.append(cargo_copy)

        return new_container

    def __repr__(self):
        return f"Container(dims=({self.length}, {self.width}, {self.height}), space_util={self.space_utilization():.1f}%, weight={self.current_weight}/{self.max_weight})"
