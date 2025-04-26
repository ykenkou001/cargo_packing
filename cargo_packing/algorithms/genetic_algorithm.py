"""
遺伝的アルゴリズムによる荷物パッキング最適化
"""

import copy
import random
import time
from typing import Any, Dict, List

import numpy as np

from cargo_packing.cargo import Cargo
from cargo_packing.container import Container


class GeneticAlgorithmPacking:
    """遺伝的アルゴリズムによる積付け最適化"""

    def __init__(
        self,
        container: Container,
        cargo_list: List[Cargo],
        population_size: int = 30,
        generations: int = 50,
    ):
        self.container_template = container
        self.cargo_list = cargo_list
        self.population_size = population_size
        self.generations = generations
        self.population = []  # 個体の集合
        self.best_solution = None  # 最良解

    def initialize_population(self):
        """初期集団を生成"""
        self.population = []

        for _ in range(self.population_size):
            # 積載順序と回転状態をランダムに決定
            individual = {
                "loading_sequence": np.random.permutation(
                    len(self.cargo_list)
                ).tolist(),
                "rotations": [
                    (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
                    for _ in range(len(self.cargo_list))
                ],
                "fitness": 0,
                "container": None,
            }
            self.population.append(individual)

    def evaluate_fitness(self, individual: Dict[str, Any]) -> float:
        """個体の適応度を評価"""
        # コンテナの複製を作成
        container = copy.deepcopy(self.container_template)

        # 指定された順序で荷物を積載
        for idx in individual["loading_sequence"]:
            cargo = copy.deepcopy(self.cargo_list[idx])

            # 回転を適用
            if cargo.rotatable:
                rx, ry, rz = individual["rotations"][idx]
                cargo.rotate(rx, ry, rz)

            # 最適な配置位置を探索
            best_pos = None
            best_z = float("inf")

            # 簡略版の探索（効率化のため）
            z_step = 1.0
            y_step = 1.0
            x_step = 1.0

            for z in np.arange(0, container.height - cargo.height + 0.1, z_step):
                for y in np.arange(0, container.width - cargo.width + 0.1, y_step):
                    for x in np.arange(
                        0, container.length - cargo.length + 0.1, x_step
                    ):
                        if container.is_position_valid(x, y, z, cargo):
                            if z < best_z:
                                best_z = z
                                best_pos = (x, y, z)
                                break
                    if best_pos and best_z == 0:
                        break
                if best_pos and best_z == 0:
                    break

            # 有効な配置位置が見つかった場合、荷物を積載
            if best_pos:
                container.load_cargo(cargo, best_pos)

        # 適応度を計算（空間利用率が高く、重量バランスが良いほど高評価）
        space_util = container.space_utilization()
        weight_balance = 100 - container.weight_balance_score() * 100  # 100が最良

        # 空間利用率と重量バランスを組み合わせた適応度
        fitness = space_util * 0.8 + weight_balance * 0.2

        individual["fitness"] = fitness
        individual["container"] = container

        return fitness

    def select_parents(self) -> List[Dict[str, Any]]:
        """親選択（トーナメント選択）"""
        parents = []
        tournament_size = max(2, self.population_size // 5)

        for _ in range(self.population_size):
            # トーナメント参加者をランダム選出
            tournament = random.sample(self.population, tournament_size)
            # 最も適応度の高い個体を選択
            winner = max(tournament, key=lambda ind: ind["fitness"])
            parents.append(winner)

        return parents

    def crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """交叉（順序交叉）"""
        # 積載順序の交叉
        seq_len = len(parent1["loading_sequence"])
        start, end = sorted(random.sample(range(seq_len), 2))

        # 親1から一部を引き継ぐ
        child_seq = [-1] * seq_len
        for i in range(start, end + 1):
            child_seq[i] = parent1["loading_sequence"][i]

        # 親2から残りを引き継ぐ
        idx = 0
        for i in range(seq_len):
            if child_seq[i] == -1:
                while parent2["loading_sequence"][idx] in child_seq:
                    idx += 1
                child_seq[i] = parent2["loading_sequence"][idx]
                idx += 1

        # 回転パターンの交叉（一様交叉）
        child_rot = []
        for i in range(seq_len):
            if random.random() < 0.5:
                child_rot.append(parent1["rotations"][i])
            else:
                child_rot.append(parent2["rotations"][i])

        child = {
            "loading_sequence": child_seq,
            "rotations": child_rot,
            "fitness": 0,
            "container": None,
        }

        return child

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """突然変異"""
        # 積載順序の突然変異（2点交換）
        seq = individual["loading_sequence"].copy()
        if random.random() < 0.3:  # 30%の確率で発生
            idx1, idx2 = random.sample(range(len(seq)), 2)
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]

        # 回転パターンの突然変異
        rot = individual["rotations"].copy()
        for i in range(len(rot)):
            if random.random() < 0.1:  # 10%の確率で発生
                rx, ry, rz = rot[i]
                # いずれかの軸を回転
                axis = random.randint(0, 2)
                if axis == 0:
                    rx = (rx + 1) % 2
                elif axis == 1:
                    ry = (ry + 1) % 2
                else:
                    rz = (rz + 1) % 2
                rot[i] = (rx, ry, rz)

        mutated = {
            "loading_sequence": seq,
            "rotations": rot,
            "fitness": 0,
            "container": None,
        }

        return mutated

    def evolve(self):
        """遺伝的アルゴリズムの主要ループ"""
        # 初期集団の生成
        self.initialize_population()

        # 各個体の適応度を評価
        for individual in self.population:
            self.evaluate_fitness(individual)

        # 最良個体を記録
        self.best_solution = max(self.population, key=lambda ind: ind["fitness"])

        start_time = time.time()

        # 世代を繰り返す
        for generation in range(self.generations):
            # 親選択
            parents = self.select_parents()

            # 新しい集団を生成
            new_population = []

            # エリート保存戦略：最良個体を次世代に残す
            elite_count = max(1, self.population_size // 10)
            sorted_pop = sorted(
                self.population, key=lambda ind: ind["fitness"], reverse=True
            )
            new_population.extend(sorted_pop[:elite_count])

            # 残りの個体を交叉と突然変異で生成
            while len(new_population) < self.population_size:
                # 親をランダム選択
                parent1, parent2 = random.sample(parents, 2)

                # 交叉
                child = self.crossover(parent1, parent2)

                # 突然変異
                child = self.mutate(child)

                # 評価
                self.evaluate_fitness(child)

                # 新集団に追加
                new_population.append(child)

            # 集団を更新
            self.population = new_population

            # 現世代の最良個体を確認
            current_best = max(self.population, key=lambda ind: ind["fitness"])
            if current_best["fitness"] > self.best_solution["fitness"]:
                self.best_solution = current_best

            # 経過報告（10世代ごと）
            if generation % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Generation {generation}: Best fitness = {self.best_solution['fitness']:.2f}, "
                    + f"Space utilization = {self.best_solution['container'].space_utilization():.1f}%, "
                    + f"Elapsed time: {elapsed:.1f}s"
                )

        print(
            f"Genetic algorithm completed. Best fitness: {self.best_solution['fitness']:.2f}"
        )
        return self.best_solution["container"]


def genetic_algorithm_loading(
    container: Container, cargo_list: List[Cargo]
) -> Container:
    """遺伝的アルゴリズムによる積付け最適化のラッパー関数"""
    ga = GeneticAlgorithmPacking(
        container, cargo_list, population_size=30, generations=50
    )
    optimized_container = ga.evolve()
    return optimized_container
