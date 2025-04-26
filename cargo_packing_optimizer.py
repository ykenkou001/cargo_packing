"""
荷物パッキング最適化のメインスクリプト
設定ファイル（YAMLまたはJSON）を使用して動作をコントロールできます
"""

import argparse
import sys

from cargo_packing.algorithms.advanced_algorithms import (
    combined_optimization_loading,
    rotation_enhanced_loading,
    weight_balanced_3d_loading,
)
from cargo_packing.algorithms.basic_algorithms import (
    layer_based_3d_loading,
    multi_layer_3d_loading,
)
from cargo_packing.container import Container
from cargo_packing.utils import (
    create_seed_cargo_csv,
    get_cargo_list,
    load_config,
    save_config_template,
)
from cargo_packing.visualization import (
    compare_algorithms_3d,
    compare_all_algorithms,
    visualize_2d_loading,
    visualize_loading,
    visualize_weight_distribution,
)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="荷物パッキング最適化プログラム")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="設定ファイルのパス（デフォルト: config.yml）",
    )
    parser.add_argument(
        "--create-config", action="store_true", help="設定ファイルのテンプレートを作成"
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="サンプルの荷物データCSVファイルを作成",
    )

    return parser.parse_args()


def main():
    # コマンドライン引数の解析
    args = parse_args()

    # テンプレート設定ファイルの作成
    if args.create_config:
        save_config_template("config_template.yml")
        print(
            "テンプレート設定ファイル（YAML形式）を作成しました。必要に応じて編集して使用してください。"
        )
        if not args.create_sample_data:
            return

    # サンプルデータの作成
    if args.create_sample_data:
        create_seed_cargo_csv("sample_cargo.csv")
        print(
            "サンプルの荷物データを作成しました。必要に応じて編集して使用してください。"
        )
        return

    # 設定ファイルの読み込み
    config = load_config(args.config)

    # 設定から荷物リストを取得
    cargo_list = get_cargo_list(config)
    if not cargo_list:
        print("荷物リストが空です。設定を確認してください。")
        return

    print(f"読み込んだ荷物の数: {len(cargo_list)}個")

    # コンテナの設定を取得
    container_config = config["container"]
    container_length = container_config["length"]
    container_width = container_config["width"]
    container_height = container_config["height"]
    max_weight = container_config["max_weight"]

    print(
        f"コンテナのサイズ: {container_length} x {container_width} x {container_height}"
    )
    print(f"最大積載重量: {max_weight}")

    # アルゴリズムの種類に基づいて処理を分岐
    algorithm_type = config["algorithm"]["type"].lower()

    if algorithm_type == "all":
        # 全アルゴリズムを比較
        compare_all_algorithms(
            cargo_list,
            container_length,
            container_width,
            container_height,
            max_weight,
        )
    elif algorithm_type == "basic":
        # 基本的な3Dアルゴリズムを比較
        compare_algorithms_3d(
            cargo_list,
            container_length,
            container_width,
            container_height,
            max_weight,
        )
    else:
        # 単一のアルゴリズムを実行
        container = Container(
            container_length, container_width, container_height, max_weight
        )

        if algorithm_type == "layer_based":
            result_container = layer_based_3d_loading(container, cargo_list)
            title = "Layer-Based 3D Algorithm"
        elif algorithm_type == "multi_layer":
            result_container = multi_layer_3d_loading(container, cargo_list)
            title = "Multi-Layer 3D Algorithm"
        elif algorithm_type == "rotation":
            result_container = rotation_enhanced_loading(container, cargo_list)
            title = "Rotation-Enhanced Algorithm"
        elif algorithm_type == "weight_balanced":
            result_container = weight_balanced_3d_loading(container, cargo_list)
            title = "Weight-Balanced 3D Algorithm"
        elif algorithm_type == "combined":
            result_container = combined_optimization_loading(container, cargo_list)
            title = "Combined Optimization Algorithm"
        elif algorithm_type == "genetic":
            # 遺伝的アルゴリズムのパラメータを取得
            genetic_params = config["algorithm"]["params"].get("genetic", {})
            population_size = genetic_params.get("population_size", 30)
            generations = genetic_params.get("generations", 50)

            # 更新したパラメータでコンテナを作成
            container = Container(
                container_length, container_width, container_height, max_weight
            )

            # 遺伝的アルゴリズムを実行
            from cargo_packing.algorithms.genetic_algorithm import (
                GeneticAlgorithmPacking,
            )

            ga = GeneticAlgorithmPacking(
                container,
                cargo_list,
                population_size=population_size,
                generations=generations,
            )
            result_container = ga.evolve()
            title = "Genetic Algorithm"
        else:
            print(f"不明なアルゴリズムタイプ: {algorithm_type}")
            print(
                "有効なタイプ: all, basic, layer_based, multi_layer, rotation, weight_balanced, combined, genetic"
            )
            return

        # 可視化オプションを取得
        vis_config = config["visualization"]
        if vis_config.get("show_3d", True):
            visualize_loading(result_container, f"{title}")
        if vis_config.get("show_2d", True):
            visualize_2d_loading(result_container, f"{title} (Side View)")
        if vis_config.get("show_weight_distribution", True):
            visualize_weight_distribution(
                result_container, f"{title} Weight Distribution"
            )

        # 結果を表示
        print(f"\nResults for {title}:")
        print(f"Space Utilization: {result_container.space_utilization():.1f}%")
        print(f"Weight Utilization: {result_container.weight_utilization():.1f}%")
        print(f"Weight Balance Score: {result_container.weight_balance_score():.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
        traceback.print_exc()
        sys.exit(1)
