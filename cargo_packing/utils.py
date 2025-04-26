"""
ユーティリティ関数を提供するモジュール
特に設定ファイルの読み込みと処理を担当
"""

import json
import os
import random
from typing import Any, Dict, List

import yaml  # YAML形式の設定ファイルのサポートを追加

from cargo_packing.cargo import Cargo


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス（YAMLまたはJSONフォーマット）

    Returns:
        設定情報を含む辞書
    """
    # デフォルト設定
    default_config = {
        "container": {
            "length": 20.0,
            "width": 10.0,
            "height": 10.0,
            "max_weight": 1000.0,
        },
        "algorithm": {
            "type": "all",
            "params": {"genetic": {"population_size": 30, "generations": 50}},
        },
        "visualization": {
            "show_3d": True,
            "show_2d": True,
            "show_weight_distribution": True,
        },
        "cargo_data": {
            "path": "",
            "random_generation": {
                "count": 15,
                "min_size": 1.0,
                "max_size": 5.0,
                "min_weight": 1.0,
                "max_weight": 100.0,
            },
        },
    }

    if not os.path.exists(config_path):
        return default_config

    # ファイル拡張子からフォーマットを判定
    _, ext = os.path.splitext(config_path)

    try:
        with open(config_path, "r") as f:
            if ext.lower() in [".yml", ".yaml"]:
                config = yaml.safe_load(f)
            else:  # '.json'など
                config = json.load(f)
    except Exception as e:
        print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
        print("デフォルト設定を使用します。")
        return default_config

    return config


def generate_random_cargo(config: Dict[str, Any]) -> List[Cargo]:
    """設定に基づいてランダムな荷物リストを生成

    Args:
        config: 設定情報を含む辞書

    Returns:
        生成された荷物のリスト
    """
    cargo_list = []
    random_cfg = config["cargo_data"]["random_generation"]

    count = random_cfg["count"]
    min_size = random_cfg["min_size"]
    max_size = random_cfg["max_size"]
    min_weight = random_cfg["min_weight"]
    max_weight = random_cfg["max_weight"]

    for i in range(count):
        # ランダムなサイズと重量を生成
        length = random.uniform(min_size, max_size)
        width = random.uniform(min_size, max_size)
        height = random.uniform(min_size, max_size)
        weight = random.uniform(min_weight, max_weight)

        # 重量の大きい荷物ほど優先度を高くする
        priority = int(weight / max_weight * 10)

        # 90%の確率で回転可能とする
        rotatable = random.random() < 0.9

        cargo = Cargo(
            id=i + 1,
            length=length,
            width=width,
            height=height,
            weight=weight,
            priority=priority,
            rotatable=rotatable,
        )
        cargo_list.append(cargo)

    return cargo_list


def load_cargo_from_file(file_path: str) -> List[Cargo]:
    """CSVファイルから荷物データを読み込む

    Args:
        file_path: CSVファイルのパス

    Returns:
        読み込まれた荷物のリスト
    """
    import csv

    cargo_list = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                cargo = Cargo(
                    id=i + 1,
                    length=float(row["length"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    weight=float(row.get("weight", 0)),
                    priority=int(row.get("priority", 0)),
                    rotatable=row.get("rotatable", "true").lower() == "true",
                )
                cargo_list.append(cargo)
            except (ValueError, KeyError) as e:
                print(f"Error loading cargo from row {i + 1}: {e}")

    return cargo_list


def get_cargo_list(config: Dict[str, Any]) -> List[Cargo]:
    """設定に基づいて荷物リストを取得

    ファイルから読み込むか、ランダムに生成するかを設定に基づいて決定

    Args:
        config: 設定情報を含む辞書

    Returns:
        荷物のリスト
    """
    file_path = config["cargo_data"].get("path", "")

    if file_path and os.path.exists(file_path):
        return load_cargo_from_file(file_path)
    else:
        return generate_random_cargo(config)


def save_config_template(file_path: str = "config_template.yml") -> None:
    """設定ファイルのテンプレートを保存

    Args:
        file_path: 保存先のファイルパス
    """
    config = {
        "container": {
            "length": 20.0,
            "width": 10.0,
            "height": 10.0,
            "max_weight": 1000.0,
        },
        "algorithm": {
            "type": "all",  # 'all', 'basic', 'advanced', 'genetic', 'combined'
            "params": {"genetic": {"population_size": 30, "generations": 50}},
        },
        "visualization": {
            "show_3d": True,
            "show_2d": True,
            "show_weight_distribution": True,
        },
        "cargo_data": {
            "path": "",  # 空文字列ならランダム生成、それ以外ならファイルから読み込み
            "random_generation": {
                "count": 15,
                "min_size": 1.0,
                "max_size": 5.0,
                "min_weight": 1.0,
                "max_weight": 100.0,
            },
        },
    }

    # ファイル拡張子からフォーマットを判定
    _, ext = os.path.splitext(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        if ext.lower() in [".yml", ".yaml"]:
            yaml.dump(
                config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        else:  # '.json'など
            json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"設定ファイルのテンプレートを {file_path} に保存しました")


def create_seed_cargo_csv(file_path: str = "sample_cargo.csv") -> None:
    """サンプルの荷物データCSVファイルを作成

    Args:
        file_path: 保存先のファイルパス
    """
    import csv

    # サンプルの荷物データ
    data = [
        {
            "length": 4.0,
            "width": 3.0,
            "height": 2.0,
            "weight": 80.0,
            "priority": 8,
            "rotatable": "true",
        },
        {
            "length": 3.0,
            "width": 2.0,
            "height": 1.0,
            "weight": 30.0,
            "priority": 5,
            "rotatable": "true",
        },
        {
            "length": 2.0,
            "width": 2.0,
            "height": 2.0,
            "weight": 40.0,
            "priority": 6,
            "rotatable": "true",
        },
        {
            "length": 5.0,
            "width": 4.0,
            "height": 3.0,
            "weight": 120.0,
            "priority": 9,
            "rotatable": "false",
        },
        {
            "length": 1.5,
            "width": 1.5,
            "height": 1.5,
            "weight": 20.0,
            "priority": 4,
            "rotatable": "true",
        },
        {
            "length": 3.5,
            "width": 2.5,
            "height": 2.0,
            "weight": 70.0,
            "priority": 7,
            "rotatable": "true",
        },
        {
            "length": 2.5,
            "width": 2.0,
            "height": 1.5,
            "weight": 35.0,
            "priority": 5,
            "rotatable": "true",
        },
        {
            "length": 4.5,
            "width": 3.5,
            "height": 2.5,
            "weight": 100.0,
            "priority": 8,
            "rotatable": "false",
        },
        {
            "length": 1.0,
            "width": 1.0,
            "height": 1.0,
            "weight": 10.0,
            "priority": 3,
            "rotatable": "true",
        },
        {
            "length": 3.0,
            "width": 3.0,
            "height": 3.0,
            "weight": 90.0,
            "priority": 7,
            "rotatable": "true",
        },
        {
            "length": 2.0,
            "width": 1.5,
            "height": 1.0,
            "weight": 15.0,
            "priority": 4,
            "rotatable": "true",
        },
        {
            "length": 4.0,
            "width": 2.0,
            "height": 1.0,
            "weight": 25.0,
            "priority": 5,
            "rotatable": "true",
        },
        {
            "length": 3.0,
            "width": 1.0,
            "height": 1.0,
            "weight": 12.0,
            "priority": 3,
            "rotatable": "true",
        },
        {
            "length": 2.0,
            "width": 2.0,
            "height": 1.0,
            "weight": 18.0,
            "priority": 4,
            "rotatable": "true",
        },
        {
            "length": 5.0,
            "width": 2.0,
            "height": 2.0,
            "weight": 85.0,
            "priority": 7,
            "rotatable": "false",
        },
    ]

    # CSVファイルに書き込み
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["length", "width", "height", "weight", "priority", "rotatable"],
        )
        writer.writeheader()
        writer.writerows(data)

    print(f"サンプル荷物データを {file_path} に保存しました")
