"""
高度な荷物パッキングアルゴリズム（回転最適化、重量バランスなど）
"""

import copy
from typing import List

import numpy as np

from cargo_packing.cargo import Cargo
from cargo_packing.container import Container


def rotation_enhanced_loading(
    container: Container, cargo_list: List[Cargo]
) -> Container:
    """回転を考慮した積付けアルゴリズム"""
    # 体積の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        best_position = None
        best_utilization = -1
        best_rotation = (0, 0, 0)

        # 回転可能な荷物のみ回転を試みる
        if cargo.rotatable:
            for rx in range(2):  # 0=0度, 1=90度 (x軸周り)
                for ry in range(2):  # 0=0度, 1=90度 (y軸周り)
                    for rz in range(2):  # 0=0度, 1=90度 (z軸周り)
                        # 荷物をコピーして回転を試す
                        test_cargo = copy.deepcopy(cargo)
                        test_cargo.rotate(rx, ry, rz)

                        # 最も低く、前方に置ける場所を探す
                        for z in range(int(container.height * 10)):
                            z_val = z / 10.0
                            if z_val + test_cargo.height > container.height:
                                continue

                            for y in range(int(container.width * 10)):
                                y_val = y / 10.0
                                if y_val + test_cargo.width > container.width:
                                    continue

                                for x in range(int(container.length * 10)):
                                    x_val = x / 10.0
                                    if x_val + test_cargo.length > container.length:
                                        continue

                                    position = (x_val, y_val, z_val)

                                    # この位置が有効か確認
                                    if container.is_position_valid(
                                        x_val, y_val, z_val, test_cargo
                                    ):
                                        # 配置位置のスコアを計算（低いz、低いy、低いxが優先）
                                        position_score = (
                                            z_val * 10000 + y_val * 100 + x_val
                                        )

                                        if (
                                            best_position is None
                                            or position_score < best_utilization
                                        ):
                                            best_position = position
                                            best_utilization = position_score
                                            best_rotation = (rx, ry, rz)
                                            break

                                if best_position:
                                    break
                            if best_position:
                                break
        else:
            # 回転不可の場合は単純に最適位置を探す
            for z in range(int(container.height * 10)):
                z_val = z / 10.0
                if z_val + cargo.height > container.height:
                    continue

                for y in range(int(container.width * 10)):
                    y_val = y / 10.0
                    if y_val + cargo.width > container.width:
                        continue

                    for x in range(int(container.length * 10)):
                        x_val = x / 10.0
                        if x_val + cargo.length > container.length:
                            continue

                        position = (x_val, y_val, z_val)

                        if container.is_position_valid(x_val, y_val, z_val, cargo):
                            position_score = z_val * 10000 + y_val * 100 + x_val

                            if (
                                best_position is None
                                or position_score < best_utilization
                            ):
                                best_position = position
                                best_utilization = position_score
                                break

                    if best_position:
                        break
                if best_position:
                    break

        if best_position:
            # 最適な回転を適用
            if cargo.rotatable:
                cargo.rotate(*best_rotation)

            # 荷物を配置
            container.load_cargo(cargo, best_position)

    return container


def weight_balanced_3d_loading(
    container: Container, cargo_list: List[Cargo]
) -> Container:
    """重量バランスを最適化する3次元積付けアルゴリズム"""
    # 重い荷物から順に積む
    sorted_cargo = sorted(cargo_list, key=lambda c: c.weight, reverse=True)

    # 重心位置を追跡
    cog_x = container.length / 2
    cog_y = container.width / 2

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        best_position = None
        best_balance_score = float("inf")  # 小さいほど良い

        # 可能な配置場所を全探索
        for z in range(int(container.height * 10)):
            z_val = z / 10.0
            if z_val + cargo.height > container.height:
                continue

            for y in range(int(container.width * 10)):
                y_val = y / 10.0
                if y_val + cargo.width > container.width:
                    continue

                for x in range(int(container.length * 10)):
                    x_val = x / 10.0
                    if x_val + cargo.length > container.length:
                        continue

                    position = (x_val, y_val, z_val)

                    if container.is_position_valid(x_val, y_val, z_val, cargo):
                        # 仮想的に配置してみて重心がどう変化するか計算
                        new_total_weight = container.current_weight + cargo.weight
                        cargo_center_x = x_val + cargo.length / 2
                        cargo_center_y = y_val + cargo.width / 2

                        if new_total_weight > 0:
                            new_cog_x = (
                                container.cog_x * container.current_weight
                                + cargo_center_x * cargo.weight
                            ) / new_total_weight
                            new_cog_y = (
                                container.cog_y * container.current_weight
                                + cargo_center_y * cargo.weight
                            ) / new_total_weight

                            # 重心の理想位置（コンテナの中心）からのずれを計算
                            x_diff = abs(new_cog_x - container.length / 2)
                            y_diff = abs(new_cog_y - container.width / 2)
                            balance_score = x_diff + y_diff

                            # 物理的安定性のためにz位置が低いほうが良い
                            stability_score = z_val

                            # 総合スコア（重心バランスと安定性の両方を考慮）
                            total_score = balance_score + stability_score * 0.5

                            if total_score < best_balance_score:
                                best_balance_score = total_score
                                best_position = position

        if best_position:
            container.load_cargo(cargo, best_position)

    return container


def combined_optimization_loading(
    container: Container, cargo_list: List[Cargo]
) -> Container:
    """層ベース、回転、重量バランスを組み合わせた最適化アルゴリズム

    1. 荷物の特性に応じて最適な積載方法を選択
    2. 大型の荷物は層ベースで効率的に配置
    3. 回転可能な荷物は回転を考慮
    4. 重い荷物は重量バランスを考慮して配置
    """
    # 荷物を特性ごとに分類
    heavy_cargo = []  # 重量が大きい荷物
    large_cargo = []  # 体積が大きい荷物
    rotatable_cargo = []  # 回転可能な小型荷物
    other_cargo = []  # その他の荷物

    # 重量と体積の閾値を計算（全体の平均値を使用）
    avg_weight = sum(c.weight for c in cargo_list) / len(cargo_list)
    avg_volume = sum(c.volume() for c in cargo_list) / len(cargo_list)
    weight_threshold = avg_weight * 1.5  # 平均の1.5倍以上を「重い」と定義
    volume_threshold = avg_volume * 1.5  # 平均の1.5倍以上を「大きい」と定義

    # 荷物を分類
    for cargo in cargo_list:
        if cargo.weight > weight_threshold:
            heavy_cargo.append(cargo)
        elif cargo.volume() > volume_threshold:
            large_cargo.append(cargo)
        elif cargo.rotatable:
            rotatable_cargo.append(cargo)
        else:
            other_cargo.append(cargo)

    print("\nCargo classification:")
    print(f"Heavy cargo: {len(heavy_cargo)} items")
    print(f"Large cargo: {len(large_cargo)} items")
    print(f"Rotatable small cargo: {len(rotatable_cargo)} items")
    print(f"Other cargo: {len(other_cargo)} items")

    # ステップ1: 重い荷物を重量バランスを考慮して配置
    print("\nStep 1: Placing heavy cargo with weight balance optimization...")
    _place_with_weight_balance(container, heavy_cargo)

    # ステップ2: 大型荷物を層ベースで配置
    print("Step 2: Placing large cargo using layer-based approach...")
    _place_with_layer_based(container, large_cargo)

    # ステップ3: 回転可能な荷物を最適な回転で配置
    print("Step 3: Placing rotatable cargo with rotation optimization...")
    _place_with_rotation(container, rotatable_cargo)

    # ステップ4: 残りの荷物を空いている場所に配置
    print("Step 4: Placing remaining cargo in available spaces...")
    _place_remaining_cargo(container, other_cargo)

    # 最終的な重量バランスを最適化
    _optimize_final_balance(container)

    return container


def _place_with_weight_balance(container: Container, cargo_list: List[Cargo]) -> None:
    """重量バランスを考慮して荷物を配置"""
    # 重い順にソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.weight, reverse=True)

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        best_position = None
        best_balance_score = float("inf")
        best_rotation = (0, 0, 0)

        # 回転可能な場合は回転も考慮
        rotations = [(0, 0, 0)]
        if cargo.rotatable:
            rotations = [
                (rx, ry, rz) for rx in range(2) for ry in range(2) for rz in range(2)
            ]

        for rx, ry, rz in rotations:
            # 荷物をコピーして回転
            test_cargo = copy.deepcopy(cargo)
            if (rx, ry, rz) != (0, 0, 0) and test_cargo.rotatable:
                test_cargo.rotate(rx, ry, rz)

            # 最下層から順に探索
            for z in range(int(container.height * 10)):
                z_val = z / 10.0
                if z_val + test_cargo.height > container.height:
                    break

                # 各位置で重心バランスを計算
                for y in range(int(container.width * 10)):
                    y_val = y / 10.0
                    if y_val + test_cargo.width > container.width:
                        break

                    for x in range(int(container.length * 10)):
                        x_val = x / 10.0
                        if x_val + test_cargo.length > container.length:
                            break

                        if container.is_position_valid(x_val, y_val, z_val, test_cargo):
                            # 仮想的に荷物を配置した場合の重心を計算
                            new_total_weight = (
                                container.current_weight + test_cargo.weight
                            )
                            cargo_center_x = x_val + test_cargo.length / 2
                            cargo_center_y = y_val + test_cargo.width / 2

                            if new_total_weight > 0:
                                new_cog_x = (
                                    (container.cog_x * container.current_weight)
                                    + (cargo_center_x * test_cargo.weight)
                                ) / new_total_weight
                                new_cog_y = (
                                    (container.cog_y * container.current_weight)
                                    + (cargo_center_y * test_cargo.weight)
                                ) / new_total_weight

                                # 理想的な位置（コンテナの中心）からの距離
                                distance_from_ideal = abs(
                                    new_cog_x - container.length / 2
                                ) + abs(new_cog_y - container.width / 2)

                                # 低い位置が優先（安定性のため）
                                position_score = distance_from_ideal + (z_val * 0.5)

                                if position_score < best_balance_score:
                                    best_balance_score = position_score
                                    best_position = (x_val, y_val, z_val)
                                    best_rotation = (rx, ry, rz)

        # 最適な回転と位置に荷物を配置
        if best_position:
            if cargo.rotatable and best_rotation != (0, 0, 0):
                cargo.rotate(*best_rotation)
            container.load_cargo(cargo, best_position)


def _place_with_layer_based(container: Container, cargo_list: List[Cargo]) -> None:
    """層ベースの方法で荷物を配置"""
    # 体積の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    current_layer_height = 0.0
    current_row_width = 0.0
    current_length_pos = 0.0
    max_height_in_layer = 0.0
    max_width_in_row = 0.0

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        # 現在の行に入らない場合、新しい行へ
        if current_length_pos + cargo.length > container.length:
            current_length_pos = 0.0
            current_row_width += max_width_in_row
            max_width_in_row = 0.0

            # 現在の層に入らない場合、新しい層へ
            if current_row_width + cargo.width > container.width:
                current_row_width = 0.0
                current_layer_height += max_height_in_layer
                max_height_in_layer = 0.0

                # コンテナの高さを超える場合はスキップ
                if current_layer_height + cargo.height > container.height:
                    continue

        # 荷物を配置
        position = (current_length_pos, current_row_width, current_layer_height)
        if container.load_cargo(cargo, position):
            max_height_in_layer = max(max_height_in_layer, cargo.height)
            max_width_in_row = max(max_width_in_row, cargo.width)
            current_length_pos += cargo.length


def _place_with_rotation(container: Container, cargo_list: List[Cargo]) -> None:
    """回転最適化した配置方法"""
    # 体積の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        best_position = None
        min_height = float("inf")
        best_rotation = (0, 0, 0)

        # すべての回転を試す
        for rx in range(2):
            for ry in range(2):
                for rz in range(2):
                    test_cargo = copy.deepcopy(cargo)
                    test_cargo.rotate(rx, ry, rz)

                    # 利用可能な最も低い位置を探す
                    for z in range(int(container.height * 10)):
                        z_val = z / 10.0
                        if z_val + test_cargo.height > container.height:
                            break

                        for y in range(int(container.width * 10)):
                            y_val = y / 10.0
                            if y_val + test_cargo.width > container.width:
                                break

                            for x in range(int(container.length * 10)):
                                x_val = x / 10.0
                                if x_val + test_cargo.length > container.length:
                                    break

                                if container.is_position_valid(
                                    x_val, y_val, z_val, test_cargo
                                ):
                                    # 最も低い位置を選択
                                    if z_val < min_height:
                                        min_height = z_val
                                        best_position = (x_val, y_val, z_val)
                                        best_rotation = (rx, ry, rz)
                                        break

                            # 最も低い位置が見つかれば次の回転を試す
                            if z_val == min_height and min_height == 0:
                                break

                        # 最も低い位置が見つかれば次の回転を試す
                        if z_val == min_height and min_height == 0:
                            break

        # 最適な回転と位置に配置
        if best_position:
            cargo.rotate(*best_rotation)
            container.load_cargo(cargo, best_position)


def _place_remaining_cargo(container: Container, cargo_list: List[Cargo]) -> None:
    """残りの荷物を配置"""
    # 体積の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    for cargo in sorted_cargo:
        if not container.can_load(cargo):
            continue

        # 最適な位置を探す
        position = container.find_valid_position(cargo)
        if position:
            container.load_cargo(cargo, position)


def _optimize_final_balance(container: Container) -> None:
    """最終的な重量バランスを最適化（小さな荷物を移動）"""
    # 現在の重量バランススコア
    current_balance = container.weight_balance_score()

    # バランスが悪い場合のみ最適化を試みる
    if current_balance < 0.05:  # 5%以下なら十分良好なバランス
        return

    print(f"Optimizing final weight balance (current score: {current_balance:.4f})...")

    # 小さな荷物を探す (全体の下位20%の体積)
    small_cargo_indices = []
    volumes = [cargo.volume() for cargo in container.cargo_list]
    volume_threshold = np.percentile(volumes, 20) if volumes else 0

    for i, cargo in enumerate(container.cargo_list):
        if cargo.volume() <= volume_threshold:
            small_cargo_indices.append(i)

    if not small_cargo_indices:
        return

    # 小さな荷物を移動して重量バランスを改善
    for idx in small_cargo_indices:
        cargo = container.cargo_list[idx]
        original_pos = cargo.position

        # コンテナから一時的に荷物を取り除く
        container.cargo_list.pop(idx)
        container.current_weight -= cargo.weight

        # 占有スペースを更新
        x1, y1, z1 = (
            int(original_pos[0] * 10),
            int(original_pos[1] * 10),
            int(original_pos[2] * 10),
        )
        x2, y2, z2 = (
            int((original_pos[0] + cargo.length) * 10),
            int((original_pos[1] + cargo.width) * 10),
            int((original_pos[2] + cargo.height) * 10),
        )
        container.space_occupied[x1:x2, y1:y2, z1:z2] = False

        # 重心を計算
        ideal_cog_x = container.length / 2
        ideal_cog_y = container.width / 2

        # 理想の重心に近づける方向を計算
        target_x = 0 if container.cog_x > ideal_cog_x else container.length
        target_y = 0 if container.cog_y > ideal_cog_y else container.width

        # ターゲット方向から探索開始
        best_pos = None
        best_score = float("inf")

        # X座標をターゲットから離れるよう探索
        x_range = (
            range(int(container.length * 10))
            if target_x == 0
            else range(int(container.length * 10) - 1, -1, -1)
        )

        for x in x_range:
            x_val = x / 10.0
            if x_val + cargo.length > container.length:
                continue

            # Y座標もターゲットから離れるよう探索
            y_range = (
                range(int(container.width * 10))
                if target_y == 0
                else range(int(container.width * 10) - 1, -1, -1)
            )

            for y in y_range:
                y_val = y / 10.0
                if y_val + cargo.width > container.width:
                    continue

                # Z座標は低い方から探索 (安定性のため)
                for z in range(int(container.height * 10)):
                    z_val = z / 10.0
                    if z_val + cargo.height > container.height:
                        break

                    if container.is_position_valid(x_val, y_val, z_val, cargo):
                        # 仮想的に配置した場合の重心を計算
                        new_total_weight = container.current_weight + cargo.weight
                        cargo_center_x = x_val + cargo.length / 2
                        cargo_center_y = y_val + cargo.width / 2

                        new_cog_x = (
                            (container.cog_x * container.current_weight)
                            + (cargo_center_x * cargo.weight)
                        ) / new_total_weight
                        new_cog_y = (
                            (container.cog_y * container.current_weight)
                            + (cargo_center_y * cargo.weight)
                        ) / new_total_weight

                        # 理想位置からの距離
                        distance = abs(new_cog_x - ideal_cog_x) + abs(
                            new_cog_y - ideal_cog_y
                        )

                        # 位置スコア (低いほど良い)
                        position_score = distance + (
                            z_val * 0.1
                        )  # 高さの影響は小さめに

                        if position_score < best_score:
                            best_score = position_score
                            best_pos = (x_val, y_val, z_val)

        if best_pos:
            # 最適な位置に配置
            container.load_cargo(cargo, best_pos)
        else:
            # 元の位置に戻す
            container.load_cargo(cargo, original_pos)

    # 最終的な重量バランスを表示
    new_balance = container.weight_balance_score()
    print(f"Final weight balance score: {new_balance:.4f} (was {current_balance:.4f})")
