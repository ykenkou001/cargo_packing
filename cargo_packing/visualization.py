"""
荷物パッキング結果の可視化モジュール
"""

import matplotlib.pyplot as plt
import numpy as np

from cargo_packing.container import Container


def visualize_loading(container: Container, title: str = "Cargo Loading Visualization"):
    """積付け結果を可視化"""
    # フォント設定 - 日本語対応
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Verdana"]
    plt.rcParams["axes.unicode_minus"] = False

    # 3Dプロット設定
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # コンテナの枠を描画
    x = [
        0,
        0,
        container.length,
        container.length,
        0,
        0,
        container.length,
        container.length,
    ]
    y = [0, container.width, container.width, 0, 0, container.width, container.width, 0]
    z = [
        0,
        0,
        0,
        0,
        container.height,
        container.height,
        container.height,
        container.height,
    ]

    # コンテナの各辺を描画
    for i, j in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], "k-", alpha=0.5)

    # 各荷物を描画
    colors = plt.cm.tab20(np.linspace(0, 1, len(container.cargo_list)))

    for i, cargo in enumerate(container.cargo_list):
        if not cargo.position:
            continue

        x, y, z = cargo.position
        dx, dy, dz = cargo.length, cargo.width, cargo.height

        # 直方体の各面を描画
        xx, yy = np.meshgrid([x, x + dx], [y, y + dy])
        ax.plot_surface(xx, yy, z * np.ones(xx.shape), color=colors[i], alpha=0.7)
        ax.plot_surface(
            xx, yy, (z + dz) * np.ones(xx.shape), color=colors[i], alpha=0.7
        )

        yy, zz = np.meshgrid([y, y + dy], [z, z + dz])
        ax.plot_surface(x * np.ones(yy.shape), yy, zz, color=colors[i], alpha=0.7)
        ax.plot_surface(
            (x + dx) * np.ones(yy.shape), yy, zz, color=colors[i], alpha=0.7
        )

        xx, zz = np.meshgrid([x, x + dx], [z, z + dz])
        ax.plot_surface(xx, y * np.ones(xx.shape), zz, color=colors[i], alpha=0.7)
        ax.plot_surface(
            xx, (y + dy) * np.ones(xx.shape), zz, color=colors[i], alpha=0.7
        )

        # ID表示
        center_x = x + dx / 2
        center_y = y + dy / 2
        center_z = z + dz / 2
        ax.text(
            center_x,
            center_y,
            center_z,
            f"ID:{cargo.id}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    ax.set_title(title)

    # 利用率情報を表示
    space_util = container.space_utilization()
    weight_util = container.weight_utilization()
    fig.text(0.02, 0.02, f"Space Utilization: {space_util:.1f}%")
    fig.text(0.5, 0.02, f"Weight Utilization: {weight_util:.1f}%")

    plt.tight_layout()
    plt.show()


def visualize_2d_loading(
    container: Container, title: str = "Cargo Loading Visualization (2D)"
):
    """積付け結果を2Dで可視化（横から見た図）"""
    # フォント設定
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Verdana"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 3))

    # コンテナを描画
    plt.axhline(
        y=0, xmin=0, xmax=container.length, color="black", linestyle="-", linewidth=2
    )
    plt.axhline(
        y=1, xmin=0, xmax=container.length, color="black", linestyle="-", linewidth=2
    )
    plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="-", linewidth=2)
    plt.axvline(
        x=container.length, ymin=0, ymax=1, color="black", linestyle="-", linewidth=2
    )

    # 各荷物を描画 (xz平面に投影)
    colors = plt.cm.tab20(np.linspace(0, 1, len(container.cargo_list)))

    for i, cargo in enumerate(container.cargo_list):
        if not cargo.position:
            continue

        x, _, z = cargo.position
        length, _, height = cargo.length, cargo.width, cargo.height

        plt.fill_between(
            [x, x + length],
            z / container.height,
            (z + height) / container.height,
            color=colors[i],
            alpha=0.7,
        )
        plt.text(
            x + length / 2,
            (z + height / 2) / container.height,
            f"ID:{cargo.id}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.title(title)
    plt.xlim(0, container.length)
    plt.ylim(0, 1)
    plt.xlabel("Position (Length)")
    plt.ylabel("Height")

    # 利用率情報を表示
    space_util = container.space_utilization()
    weight_util = container.weight_utilization()
    plt.figtext(0.02, 0.02, f"Space Utilization: {space_util:.1f}%")
    plt.figtext(0.5, 0.02, f"Weight Utilization: {weight_util:.1f}%")

    plt.tight_layout()
    plt.show()


def visualize_weight_distribution(
    container: Container, title: str = "Weight Distribution"
):
    """重量分布を可視化"""
    plt.figure(figsize=(10, 8))

    # ヒートマップで重量分布を表示
    plt.imshow(
        container.weight_distribution.T,
        origin="lower",
        extent=[0, container.length, 0, container.width],
        aspect="auto",
        cmap="hot",
    )
    plt.colorbar(label="Weight per unit area")

    # 重心位置を表示
    plt.plot(
        container.cog_x, container.cog_y, "bx", markersize=12, label="Center of Gravity"
    )
    plt.plot(
        container.length / 2,
        container.width / 2,
        "g+",
        markersize=12,
        label="Ideal Center",
    )

    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Width")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 重心バランススコアを表示
    balance_score = container.weight_balance_score()
    print(f"Weight Balance Score: {balance_score:.4f} (0=perfect, 1=worst)")
    print(f"Center of Gravity: X={container.cog_x:.2f}, Y={container.cog_y:.2f}")
    print(f"Ideal Center: X={container.length / 2:.2f}, Y={container.width / 2:.2f}")


def compare_algorithms_3d(
    cargo_list,
    container_length: float,
    container_width: float,
    container_height: float,
    max_weight: float,
):
    """3D空間での異なるアルゴリズムを比較"""
    from cargo_packing.algorithms.basic_algorithms import (
        layer_based_3d_loading,
        multi_layer_3d_loading,
    )

    # Layer-based 3D loading
    container3d = Container(
        container_length, container_width, container_height, max_weight
    )
    container3d = layer_based_3d_loading(container3d, cargo_list)
    visualize_loading(container3d, "3D Layer-Based Algorithm")
    visualize_2d_loading(container3d, "3D Layer-Based Algorithm (Side View)")

    print("============= 3D Algorithm Results =============")
    print(
        f"3D Layer-Based: Space Util={container3d.space_utilization():.1f}%, "
        f"Weight Util={container3d.weight_utilization():.1f}%"
    )

    # マルチレイヤー3D配置
    container3d_multi = Container(
        container_length, container_width, container_height, max_weight
    )
    container3d_multi = multi_layer_3d_loading(container3d_multi, cargo_list)
    visualize_loading(container3d_multi, "3D Multi-Layer Algorithm")
    visualize_2d_loading(container3d_multi, "3D Multi-Layer Algorithm (Side View)")

    print("\n============= Multi-Layer 3D Algorithm Results =============")
    print(
        f"3D Multi-Layer: Space Util={container3d_multi.space_utilization():.1f}%, "
        f"Weight Util={container3d_multi.weight_utilization():.1f}%"
    )


def compare_all_algorithms(
    cargo_list,
    container_length: float,
    container_width: float,
    container_height: float,
    max_weight: float,
):
    """すべての積付けアルゴリズムを比較"""
    from cargo_packing.algorithms.advanced_algorithms import (
        combined_optimization_loading,
        rotation_enhanced_loading,
        weight_balanced_3d_loading,
    )
    from cargo_packing.algorithms.basic_algorithms import layer_based_3d_loading
    from cargo_packing.algorithms.genetic_algorithm import genetic_algorithm_loading

    print("Starting algorithm comparison...")

    # 基本的なレイヤーベースの3Dアルゴリズム
    container1 = Container(
        container_length, container_width, container_height, max_weight
    )
    print("\nExecuting layer-based 3D loading...")
    container1 = layer_based_3d_loading(container1, cargo_list)

    # 回転強化アルゴリズム
    container2 = Container(
        container_length, container_width, container_height, max_weight
    )
    print("\nExecuting rotation-enhanced loading...")
    container2 = rotation_enhanced_loading(container2, cargo_list)

    # 重量バランス最適化アルゴリズム
    container3 = Container(
        container_length, container_width, container_height, max_weight
    )
    print("\nExecuting weight-balanced 3D loading...")
    container3 = weight_balanced_3d_loading(container3, cargo_list)

    # 組み合わせアルゴリズム
    container5 = Container(
        container_length, container_width, container_height, max_weight
    )
    print("\nExecuting combined optimization loading...")
    container5 = combined_optimization_loading(container5, cargo_list)

    # 遺伝的アルゴリズム（時間がかかる場合はコメントアウト）
    container4 = Container(
        container_length, container_width, container_height, max_weight
    )
    print("\nExecuting genetic algorithm loading (this may take a while)...")
    container4 = genetic_algorithm_loading(container4, cargo_list)

    # 結果の比較と可視化
    print("\n========== Algorithm Comparison ==========")

    print("\n1. Layer-based 3D Loading:")
    print(f"   Space Utilization: {container1.space_utilization():.1f}%")
    print(f"   Weight Utilization: {container1.weight_utilization():.1f}%")
    print(f"   Weight Balance Score: {container1.weight_balance_score():.4f}")

    print("\n2. Rotation-Enhanced Loading:")
    print(f"   Space Utilization: {container2.space_utilization():.1f}%")
    print(f"   Weight Utilization: {container2.weight_utilization():.1f}%")
    print(f"   Weight Balance Score: {container2.weight_balance_score():.4f}")

    print("\n3. Weight-Balanced 3D Loading:")
    print(f"   Space Utilization: {container3.space_utilization():.1f}%")
    print(f"   Weight Utilization: {container3.weight_utilization():.1f}%")
    print(f"   Weight Balance Score: {container3.weight_balance_score():.4f}")

    print("\n4. Genetic Algorithm Loading:")
    print(f"   Space Utilization: {container4.space_utilization():.1f}%")
    print(f"   Weight Utilization: {container4.weight_utilization():.1f}%")
    print(f"   Weight Balance Score: {container4.weight_balance_score():.4f}")

    print("\n5. Combined Optimization Loading:")
    print(f"   Space Utilization: {container5.space_utilization():.1f}%")
    print(f"   Weight Utilization: {container5.weight_utilization():.1f}%")
    print(f"   Weight Balance Score: {container5.weight_balance_score():.4f}")

    # 各アルゴリズムの結果を順番に可視化
    print("\nVisualizing results...")

    # レイヤーベース
    visualize_loading(container1, "1. Layer-based 3D Algorithm")
    visualize_2d_loading(container1, "1. Layer-based 3D Algorithm (Side View)")
    visualize_weight_distribution(container1, "1. Layer-based Weight Distribution")

    # 回転強化
    visualize_loading(container2, "2. Rotation-Enhanced Algorithm")
    visualize_2d_loading(container2, "2. Rotation-Enhanced Algorithm (Side View)")
    visualize_weight_distribution(
        container2, "2. Rotation-Enhanced Weight Distribution"
    )

    # 重量バランス最適化
    visualize_loading(container3, "3. Weight-Balanced 3D Algorithm")
    visualize_2d_loading(container3, "3. Weight-Balanced 3D Algorithm (Side View)")
    visualize_weight_distribution(container3, "3. Weight-Balanced Weight Distribution")

    # 遺伝的アルゴリズム
    visualize_loading(container4, "4. Genetic Algorithm Packing")
    visualize_2d_loading(container4, "4. Genetic Algorithm Packing (Side View)")
    visualize_weight_distribution(
        container4, "4. Genetic Algorithm Weight Distribution"
    )

    # 組み合わせアルゴリズム
    visualize_loading(container5, "5. Combined Optimization Algorithm")
    visualize_2d_loading(container5, "5. Combined Optimization Algorithm (Side View)")
    visualize_weight_distribution(
        container5, "5. Combined Optimization Weight Distribution"
    )
