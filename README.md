![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-brightgreen)
![fcsparser](https://img.shields.io/badge/uses-fcsparser-orange)
[![PyPI - fcsparser](https://img.shields.io/pypi/v/fcsparser)](https://pypi.org/project/fcsparser/)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![FCS](https://img.shields.io/badge/.fcs-supported-lightgrey)
![Deployed](https://img.shields.io/badge/Deployed%20on-Streamlit%20Cloud-orange)
# FACS Data Analysis Platform

StreamlitベースのFACS（フローサイトメトリー）データ解析Webアプリケーション

## 🔬 機能概要

### 基本機能
- **FCSファイル解析**: 標準的なFCSファイルの読み込みと解析
- **データプレビュー**: 統計情報と基本的なデータ構造の表示
- **可視化**: ヒストグラム、散布図、等高線プロットの作成
- **インタラクティブ機能**: Bokehを使用した高品質な可視化とインタラクション

### 高度な機能
- **ゲーティング**: マウスクリックによる解析領域の設定
- **統計解析**: ゲート領域での詳細統計情報
- **データ変換**: 各種変換機能（asinh, log, biexponential）
- **エクスポート**: 解析結果のCSV出力

## 📋 システム要件

### Python環境
- Python 3.8以上
- Streamlit 1.28.0以上

### 依存ライブラリ
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
bokeh>=3.2.0
scipy>=1.10.0
scikit-image>=0.21.0
shapely>=2.0.0
flowkit>=0.7.0
plotly>=5.15.0
altair>=5.0.0
```

## 🚀 インストール・実行

### ローカル環境での実行

1. **リポジトリのクローン**
```bash
git clone <repository-url>
cd facs-analysis-app
```

2. **仮想環境の作成**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

4. **アプリケーションの実行**
```bash
streamlit run app.py
```

### Streamlit Cloudでのデプロイ

1. GitHubリポジトリにコードをプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
3. リポジトリを選択してデプロイ

## 📁 プロジェクト構成

```
facs-analysis-app/
├── app.py                 # メインアプリケーション
├── config.py             # 設定ファイル
├── requirements.txt      # 依存関係
├── README.md            # プロジェクト説明
├── .gitignore           # Git除外設定
├── .streamlit/
│   └── config.toml      # Streamlit設定
├── pages/
│   ├── basic_analysis.py     # 基本解析ページ
│   └── advanced_gating.py    # 高度ゲーティングページ
└── utils/
    ├── fcs_processor.py      # FCS処理ユーティリティ
    ├── plotting.py           # プロット作成機能
    └── gating.py            # ゲーティング機能
```

## 🎯 使用方法

### 基本的な解析フロー

1. **ファイルアップロード**
   - サイドバーからFCSファイルを選択・アップロード
   - ファイル情報と基本統計の確認

2. **基本解析**
   - チャンネル選択とデータプレビュー
   - ヒストグラムと散布図の作成
   - 統計情報の確認

3. **高度な解析**
   - ゲーティング領域の設定
   - 等高線プロットでの密度可視化
   - ゲート統計の計算

4. **結果のエクスポート**
   - 解析結果のCSV出力
   - プロット画像の保存

### ゲーティング機能

- **ポリゴンゲート**: マウスクリックで多角形領域を設定
- **矩形ゲート**: ドラッグで矩形領域を設定
- **楕円ゲート**: 楕円形の解析領域を設定
- **閾値ゲート**: 単一チャンネルでの閾値設定

## ⚙️ 設定カスタマイズ

### config.py での設定変更

- **UI設定**: 色、サイズ、レイアウト
- **プロット設定**: ビン数、カラーマップ、透明度
- **ゲーティング設定**: ゲート色、線幅、最小ポイント数
- **データ処理設定**: 変換方法、補正パラメータ

### Streamlit設定

`.streamlit/config.toml`でアプリケーションの動作を調整：
- アップロードサイズ制限
- テーマカラー
- パフォーマンス設定

## 🔧 開発・カスタマイズ

### 新機能の追加

1. **新しいページの追加**
   - `pages/`ディレクトリに新しいPythonファイルを作成
   - `app.py`でページを登録

2. **新しいプロット機能**
   - `utils/plotting.py`に新しい関数を追加
   - 必要に応じてBokehやPlotlyのウィジェットを実装

3. **ゲーティング機能の拡張**
   - `utils/gating.py`に新しいゲートタイプを追加
   - JavaScriptコールバックの実装

### デバッグ・ログ

開発時のデバッグには以下を活用：
```python
import streamlit as st
st.write("Debug info:", variable)  # デバッグ出力
st.json(data)  # JSONデータの表示
```

## 📊 サポートするデータ形式

- **FCS 2.0/3.0/3.1**: 標準的なフローサイトメトリーファイル
- **チャンネル**: FSC, SSC, 各種蛍光チャンネル
- **パラメータ**: リニア・ログスケール対応

## 🐛 トラブルシューティング

### よくある問題

1. **ファイルアップロードエラー**
   - ファイルサイズ制限（100MB）を確認
   - FCS形式の妥当性をチェック

2. **プロット表示問題**
   - ブラウザのJavaScript有効化を確認
   - キャッシュクリア

3. **パフォーマンス問題**
   - データサイズの縮小
   - プロットのビン数調整

### ログとエラー処理

エラーメッセージは`config.py`で定義され、適切な日本語メッセージを表示します。

## 📝 ライセンス

このプロジェクトは研究・教育目的で作成されています。商用利用については開発者にお問い合わせください。

## 🤝 貢献

バグ報告や機能要望は、GitHubのIssuesでお知らせください。プルリクエストも歓迎します。

## 📞 サポート

技術的な質問やサポートが必要な場合は、プロジェクトのメンテナーにお問い合わせください。
