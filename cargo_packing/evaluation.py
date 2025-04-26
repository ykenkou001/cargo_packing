"""
荷物パッキングアルゴリズムの性能評価モジュール
"""

import time
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# 日本語フォントのサポートを追加
# フォントが利用可能かどうかをチェックし、利用可能ならそれを使用
def setup_japanese_font():
    """日本語表示可能なフォントをセットアップ"""
    # 一般的な日本語対応フォントをリストアップ
    japanese_fonts = [
        "IPAGothic",
        "IPAPGothic",
        "IPAexGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "MS Gothic",
        "Meiryo",
        "Yu Gothic",
        "Hiragino Sans GB",
        "TakaoPGothic",
    ]

    font_found = False
    for font in japanese_fonts:
        try:
            mpl.font_manager.findfont(font)
            plt.rcParams["font.family"] = font
            print(f"日本語フォント '{font}' を使用します。")
            font_found = True
            break
        except:
            continue

    if not font_found:
        # 日本語フォントが見つからない場合は、タイトルやラベルを英語に変更
        print("適切な日本語フォントが見つかりません。英語でグラフを表示します。")
        return False

    return True


from cargo_packing.algorithms.advanced_algorithms import (  # noqa
    combined_optimization_loading,
    rotation_enhanced_loading,
    weight_balanced_3d_loading,
)
from cargo_packing.algorithms.basic_algorithms import (  # noqa
    layer_based_3d_loading,
    multi_layer_3d_loading,
)
from cargo_packing.cargo import Cargo  # noqa
from cargo_packing.container import Container  # noqa


class AlgorithmEvaluator:
    """アルゴリズム評価クラス"""

    def __init__(
        self,
        cargo_list: List[Cargo],
        container_length: float,
        container_width: float,
        container_height: float,
        max_weight: float,
    ):
        self.cargo_list = cargo_list
        self.container_length = container_length
        self.container_width = container_width
        self.container_height = container_height
        self.max_weight = max_weight
        self.results = {}

        # 評価指標の定義
        self.metrics = [
            "空間利用率(%)",
            "重量利用率(%)",
            "重量バランススコア",
            "積載率(%)",
            "実行時間(秒)",
        ]

    def create_container(self) -> Container:
        """新しいコンテナインスタンスを作成"""
        return Container(
            self.container_length,
            self.container_width,
            self.container_height,
            self.max_weight,
        )

    def evaluate_algorithm(
        self, algorithm_func, algorithm_name: str, **kwargs
    ) -> Dict[str, Any]:
        """
        単一アルゴリズムの評価を実行

        Args:
            algorithm_func: アルゴリズム関数
            algorithm_name: アルゴリズム名
            **kwargs: アルゴリズム関数に渡す追加パラメータ

        Returns:
            評価結果を含む辞書
        """
        print(f"\n{algorithm_name} の評価を開始...")

        container = self.create_container()

        # 実行時間の計測開始
        start_time = time.time()

        # アルゴリズムの実行
        result_container = algorithm_func(container, self.cargo_list.copy(), **kwargs)

        # 実行時間の計測終了
        execution_time = time.time() - start_time

        # 評価指標の計算
        space_utilization = result_container.space_utilization()
        weight_utilization = result_container.weight_utilization()
        weight_balance = result_container.weight_balance_score()
        loading_ratio = len(result_container.cargo_list) / len(self.cargo_list) * 100

        # 結果を辞書に格納
        result = {
            "空間利用率(%)": space_utilization,
            "重量利用率(%)": weight_utilization,
            "重量バランススコア": weight_balance,
            "積載率(%)": loading_ratio,
            "実行時間(秒)": execution_time,
            "コンテナ": result_container,
        }

        # クラス変数に結果を保存
        self.results[algorithm_name] = result

        print(f"{algorithm_name} の評価が完了しました。")
        print(f"- 空間利用率: {space_utilization:.2f}%")
        print(f"- 重量利用率: {weight_utilization:.2f}%")
        print(f"- 重量バランススコア: {weight_balance:.4f} (低いほど良い)")
        print(f"- 積載率: {loading_ratio:.2f}%")
        print(f"- 実行時間: {execution_time:.3f}秒")

        return result

    def evaluate_all_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        全アルゴリズムの評価を実行

        Returns:
            アルゴリズム名をキー、評価結果を値とする辞書
        """
        print("全アルゴリズムの評価を開始します...")

        # Layer Based 3D Algorithm
        self.evaluate_algorithm(layer_based_3d_loading, "Layer Based 3D Algorithm")

        # Multi-Layer 3D Algorithm
        self.evaluate_algorithm(multi_layer_3d_loading, "Multi-Layer 3D Algorithm")

        # Rotation-Enhanced Algorithm
        self.evaluate_algorithm(
            rotation_enhanced_loading, "Rotation-Enhanced Algorithm"
        )

        # Weight-Balanced 3D Algorithm
        self.evaluate_algorithm(
            weight_balanced_3d_loading, "Weight-Balanced 3D Algorithm"
        )

        # Combined Optimization Algorithm
        self.evaluate_algorithm(
            combined_optimization_loading, "Combined Optimization Algorithm"
        )

        return self.results

    def evaluate_custom_algorithms(
        self, algorithm_list: List[Tuple]
    ) -> Dict[str, Dict[str, Any]]:
        """
        指定されたアルゴリズムのリストを評価

        Args:
            algorithm_list: (アルゴリズム関数, アルゴリズム名, パラメータ辞書)のタプルのリスト

        Returns:
            アルゴリズム名をキー、評価結果を値とする辞書
        """
        print("指定されたアルゴリズムの評価を開始します...")

        for algo_func, algo_name, algo_params in algorithm_list:
            self.evaluate_algorithm(algo_func, algo_name, **algo_params)

        return self.results

    def print_comparison_table(self) -> None:
        """評価結果を表形式で出力"""
        if not self.results:
            print("評価結果がありません。まずアルゴリズムを評価してください。")
            return

        # データフレームに変換
        results_data = {}
        for algo_name, result in self.results.items():
            algo_results = {}
            for metric in self.metrics:
                algo_results[metric] = result[metric]
            results_data[algo_name] = algo_results

        df = pd.DataFrame(results_data).T

        print("\nアルゴリズム評価結果比較:\n")
        print(df.to_string(float_format="%.2f"))

        # 最良の結果を強調表示
        best_space_util = df["空間利用率(%)"].idxmax()
        best_weight_balance = df["重量バランススコア"].idxmin()
        best_loading_ratio = df["積載率(%)"].idxmax()
        best_time = df["実行時間(秒)"].idxmin()

        print("\n最良の結果:")
        print(
            f"- 最高空間利用率: {best_space_util} ({df.loc[best_space_util, '空間利用率(%)']:.2f}%)"
        )
        print(
            f"- 最良重量バランス: {best_weight_balance} ({df.loc[best_weight_balance, '重量バランススコア']:.4f})"
        )
        print(
            f"- 最高積載率: {best_loading_ratio} ({df.loc[best_loading_ratio, '積載率(%)']:.2f}%)"
        )
        print(
            f"- 最短実行時間: {best_time} ({df.loc[best_time, '実行時間(秒)']:.3f}秒)"
        )

    def export_results_to_csv(self, filepath: str = "algorithm_comparison.csv") -> None:
        """評価結果をCSVファイルに出力

        Args:
            filepath: 出力先ファイルパス
        """
        if not self.results:
            print("評価結果がありません。まずアルゴリズムを評価してください。")
            return

        # データフレームに変換
        results_data = {}
        for algo_name, result in self.results.items():
            algo_results = {}
            for metric in self.metrics:
                algo_results[metric] = result[metric]
            results_data[algo_name] = algo_results

        df = pd.DataFrame(results_data).T

        # CSVに保存
        df.to_csv(filepath)
        print(f"評価結果を {filepath} に保存しました。")

    def visualize_comparison(self, save_path: str = None) -> None:
        """評価結果を視覚化

        Args:
            save_path: 画像保存先パス（Noneの場合は保存せず表示のみ）
        """
        if not self.results:
            print("評価結果がありません。まずアルゴリズムを評価してください。")
            return

        # データフレームに変換
        results_data = {}
        for algo_name, result in self.results.items():
            algo_results = {}
            for metric in self.metrics:
                if metric != "コンテナ":  # コンテナオブジェクトは除く
                    algo_results[metric] = result[metric]
            results_data[algo_name] = algo_results

        df = pd.DataFrame(results_data).T

        # 日本語フォントのセットアップ
        use_japanese = setup_japanese_font()

        # タイトルとラベルのテキスト（日本語/英語）
        titles = {
            "main": "アルゴリズム性能比較"
            if use_japanese
            else "Algorithm Performance Comparison",
            "space_weight": "空間・重量利用率"
            if use_japanese
            else "Space & Weight Utilization",
            "balance": "重量バランススコア（低いほど良い）"
            if use_japanese
            else "Weight Balance Score (Lower is Better)",
            "loading": "積載率" if use_japanese else "Loading Ratio",
            "time": "実行時間" if use_japanese else "Execution Time",
        }

        # グラフ描画設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(titles["main"], fontsize=16)

        # 空間利用率と重量利用率のグラフ（棒グラフ）
        ax1 = axes[0, 0]
        cols = (
            ["空間利用率(%)", "重量利用率(%)"]
            if use_japanese
            else ["Space Util. (%)", "Weight Util. (%)"]
        )
        # 日本語列名を英語に変換してプロット
        plotting_df = df.copy()
        if not use_japanese:
            plotting_df = plotting_df.rename(
                columns={
                    "空間利用率(%)": "Space Util. (%)",
                    "重量利用率(%)": "Weight Util. (%)",
                    "重量バランススコア": "Weight Balance",
                    "積載率(%)": "Loading Ratio (%)",
                    "実行時間(秒)": "Time (sec)",
                }
            )
            cols = ["Space Util. (%)", "Weight Util. (%)"]

        plotting_df[cols].plot(kind="bar", ax=ax1)
        ax1.set_title(titles["space_weight"])
        ax1.set_ylabel("Utilization (%)" if not use_japanese else "利用率 (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax1.legend(loc="best")

        # 重量バランススコアのグラフ（低いほど良い）
        ax2 = axes[0, 1]
        balance_col = "重量バランススコア" if use_japanese else "Weight Balance"
        plotting_df[balance_col].plot(kind="bar", ax=ax2, color="orange")
        ax2.set_title(titles["balance"])
        ax2.set_ylabel("Score" if not use_japanese else "スコア")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # 積載率のグラフ
        ax3 = axes[1, 0]
        loading_col = "積載率(%)" if use_japanese else "Loading Ratio (%)"
        plotting_df[loading_col].plot(kind="bar", ax=ax3, color="green")
        ax3.set_title(titles["loading"])
        ax3.set_ylabel("Ratio (%)" if not use_japanese else "積載率 (%)")
        ax3.set_ylim(0, 100)
        ax3.yaxis.set_major_formatter(PercentFormatter())
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # 実行時間のグラフ（低いほど良い）
        ax4 = axes[1, 1]
        time_col = "実行時間(秒)" if use_japanese else "Time (sec)"
        plotting_df[time_col].plot(kind="bar", ax=ax4, color="red")
        ax4.set_title(titles["time"])
        ax4.set_ylabel("Time (sec)" if not use_japanese else "時間 (秒)")
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存または表示
        if save_path:
            plt.savefig(save_path)
            print(f"比較グラフを {save_path} に保存しました。")
        else:
            plt.show()


def compare_algorithm_performance(
    cargo_list: List[Cargo],
    container_length: float,
    container_width: float,
    container_height: float,
    max_weight: float,
    export_csv: bool = True,
    visualize: bool = True,
    visualization_path: str = None,
) -> Dict[str, Dict[str, Any]]:
    """
    複数のアルゴリズムのパフォーマンスを比較して結果を返す

    Args:
        cargo_list: 荷物リスト
        container_length: コンテナの長さ
        container_width: コンテナの幅
        container_height: コンテナの高さ
        max_weight: コンテナの最大重量
        export_csv: CSVに結果を出力するかどうか
        visualize: 結果をグラフで視覚化するかどうか
        visualization_path: 視覚化結果の保存パス

    Returns:
        アルゴリズムごとの評価結果辞書
    """
    evaluator = AlgorithmEvaluator(
        cargo_list, container_length, container_width, container_height, max_weight
    )

    # 全アルゴリズムの評価
    results = evaluator.evaluate_all_algorithms()

    # 結果の表示
    evaluator.print_comparison_table()

    # CSVに出力（オプション）
    if export_csv:
        evaluator.export_results_to_csv()

    # グラフ表示（オプション）
    if visualize:
        evaluator.visualize_comparison(visualization_path)

    return results
    return results
