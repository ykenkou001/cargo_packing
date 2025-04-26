"""
基本的な荷物パッキングアルゴリズム
"""

from typing import List

from cargo_packing.cargo import Cargo
from cargo_packing.container import Container


def first_fit_decreasing(container: Container, cargo_list: List[Cargo]) -> Container:
    """First Fit Decreasing アルゴリズムによる積付け"""
    # 荷物を長さの降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.length, reverse=True)

    position = 0.0
    for cargo in sorted_cargo:
        if container.can_load(cargo):
            container.load_cargo(cargo, (position, 0.0, 0.0))
            position += cargo.length

    return container


def weight_balanced_loading(container: Container, cargo_list: List[Cargo]) -> Container:
    """重量バランスを考慮した積付けアルゴリズム"""
    # 荷物を重量の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.weight, reverse=True)

    # 重い荷物を前後交互に配置
    front_position = 0.0
    back_position = container.length

    for i, cargo in enumerate(sorted_cargo):
        if not container.can_load(cargo):
            continue

        if i % 2 == 0:  # 前側に配置
            container.load_cargo(cargo, (front_position, 0.0, 0.0))
            front_position += cargo.length
        else:  # 後ろ側に配置
            back_position -= cargo.length
            container.load_cargo(cargo, (back_position, 0.0, 0.0))

    return container


def priority_based_loading(container: Container, cargo_list: List[Cargo]) -> Container:
    """優先度を考慮した積付けアルゴリズム"""
    # 荷物を優先度の降順でソート
    sorted_cargo = sorted(
        cargo_list, key=lambda c: (c.priority, c.length), reverse=True
    )

    position = 0.0
    for cargo in sorted_cargo:
        if container.can_load(cargo):
            container.load_cargo(cargo, (position, 0.0, 0.0))
            position += cargo.length

    return container


def layer_based_3d_loading(container: Container, cargo_list: List[Cargo]) -> Container:
    """層ごとの3次元積付けアルゴリズム"""
    # 体積の降順でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    # 単純なシェルフアルゴリズム
    current_layer_height = 0.0
    current_row_width = 0.0
    current_length_pos = 0.0

    max_height_in_layer = 0.0
    max_width_in_row = 0.0

    # 荷物の配置を層ごとに分類するための辞書
    layers = {}

    for cargo in sorted_cargo:
        # コンテナに入らない場合はスキップ
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

                print(f"Moving to new layer at height: {current_layer_height}")

                # コンテナの高さを超える場合はスキップ
                if current_layer_height + cargo.height > container.height:
                    continue

        # 荷物を配置
        position = (current_length_pos, current_row_width, current_layer_height)
        success = container.load_cargo(cargo, position)

        if success:
            # 現在の層に荷物を追加
            layer_num = int(current_layer_height)
            if layer_num not in layers:
                layers[layer_num] = []
            layers[layer_num].append(cargo)

            # 現在の行と層の最大高さ/幅を更新
            max_height_in_layer = max(max_height_in_layer, cargo.height)
            max_width_in_row = max(max_width_in_row, cargo.width)
            current_length_pos += cargo.length

    # 各層の情報を表示
    print("\n==== Layer Information ====")
    total_loaded = 0
    for layer_num, cargos in sorted(layers.items()):
        print(f"Layer {layer_num}: {len(cargos)} items")
        total_loaded += len(cargos)
    print(f"Total loaded items: {total_loaded}/{len(sorted_cargo)}")
    print(f"Total layers used: {len(layers)}")

    return container


def multi_layer_3d_loading(container: Container, cargo_list: List[Cargo]) -> Container:
    """複数層を強制的に使用する3次元積付けアルゴリズム

    荷物をより多くの層に分散させ、視覚的に複数層の積み付けが確認しやすくなります。
    """
    # 荷物を体積でソート
    sorted_cargo = sorted(cargo_list, key=lambda c: c.volume(), reverse=True)

    # 荷物グループ分け (大、中、小)
    large_cargo = sorted_cargo[: len(sorted_cargo) // 3]
    medium_cargo = sorted_cargo[len(sorted_cargo) // 3 : 2 * len(sorted_cargo) // 3]
    small_cargo = sorted_cargo[2 * len(sorted_cargo) // 3 :]

    # 層ごとに配置する高さ
    layer_heights = [0.0, 4.0, 7.0]  # 底面、中間、上部の3層

    # 大きい荷物は底面に配置
    current_x = 0.0
    current_y = 0.0
    for cargo in large_cargo:
        if not container.can_load(cargo):
            continue

        # 行が埋まったら次の行へ
        if current_x + cargo.length > container.length:
            current_x = 0.0
            current_y += 3.0  # 行間隔を設定

            # 幅を超える場合はこの層には配置できない
            if current_y + cargo.width > container.width:
                break

        # 配置
        success = container.load_cargo(cargo, (current_x, current_y, layer_heights[0]))
        if success:
            current_x += cargo.length

    # 中サイズの荷物は中間層に配置
    current_x = 0.0
    current_y = 0.0
    for cargo in medium_cargo:
        if not container.can_load(cargo):
            continue

        # 行が埋まったら次の行へ
        if current_x + cargo.length > container.length:
            current_x = 0.0
            current_y += 2.5  # 行間隔を設定

            # 幅を超える場合はこの層には配置できない
            if current_y + cargo.width > container.width:
                break

        # 配置
        success = container.load_cargo(cargo, (current_x, current_y, layer_heights[1]))
        if success:
            current_x += cargo.length

    # 小さい荷物は上部層に配置
    current_x = 0.0
    current_y = 0.0
    for cargo in small_cargo:
        if not container.can_load(cargo):
            continue

        # 行が埋まったら次の行へ
        if current_x + cargo.length > container.length:
            current_x = 0.0
            current_y += 2.0  # 行間隔を設定

            # 幅を超える場合はこの層には配置できない
            if current_y + cargo.width > container.width:
                break

        # 配置
        success = container.load_cargo(cargo, (current_x, current_y, layer_heights[2]))
        if success:
            current_x += cargo.length

    # 各層の情報を表示
    layers = {0: [], 1: [], 2: []}
    for cargo in container.cargo_list:
        if cargo.position[2] == layer_heights[0]:
            layers[0].append(cargo)
        elif cargo.position[2] == layer_heights[1]:
            layers[1].append(cargo)
        elif cargo.position[2] == layer_heights[2]:
            layers[2].append(cargo)

    print("\n==== Multi-Layer Information ====")
    total_loaded = 0
    for layer_num, cargos in sorted(layers.items()):
        print(f"Layer {layer_num}: {len(cargos)} items")
        total_loaded += len(cargos)
    print(f"Total loaded items: {total_loaded}/{len(sorted_cargo)}")

    return container
