# FACS Data Analysis

StreamlitベースのFACS（フローサイトメトリー）データ解析Webアプリケーション

## 📋 概要

FACS Data Analysisは、フローサイトメトリーデータ（FCSファイル）を解析するためのWebベースのアプリケーションです。研究者や技術者が直感的なインターフェースでデータを可視化・解析できるよう設計されています。

## ✨ 主な機能

### データ処理
- **FCSファイル解析**: 標準的なFCSファイル（2.0/3.0/3.1）の読み込みと解析
- **データプレビュー**: 統計情報と基本的なデータ構造の表示
- **データ変換**: 各種変換機能（asinh, log, biexponential）

### 可視化機能
- **ヒストグラム**: 各チャンネルの分布表示
- **散布図**: 2Dプロットでのデータ可視化
- **等高線プロット**: 密度可視化による詳細解析
- **インタラクティブ機能**: Bokehを使用した高品質な可視化とインタラクション

### ゲーティング機能
- **ポリゴンゲート**: マウスクリックで多角形領域を設定
- **矩形ゲート**: ドラッグで矩形領域を設定
- **楕円ゲート**: 楕円形の解析領域を設定
- **閾値ゲート**: 単一チャンネルでの閾値設定

### 統計解析
- **基本統計**: 平均値、中央値、標準偏差などの算出
- **ゲート統計**: ゲート領域での詳細統計情報
- **データエクスポート**: 解析結果のCSV出力

## 🔧 システム要件

- Python 3.8以上
- Streamlit 1.28.0以上

## 📦 依存関係

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

## 🚀 インストールと実行

### ローカル環境での実行

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/ymmtshch/FACSDataAnalysis.git
   cd FACSDataAnalysis
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

## 📁 プロジェクト構造

```
facs-analysis-app/
├── app.py                    # メインアプリケーション
├── config.py                 # 設定ファイル
├── requirements.txt          # 依存関係
├── README.md                 # プロジェクト説明
├── .gitignore               # Git除外設定
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── pages/
│   ├── basic_analysis.py    # 基本解析ページ
│   └── advanced_gating.py   # 高度ゲーティングページ
└── utils/
    ├── fcs_processor.py     # FCS処理ユーティリティ
    ├── plotting.py          # プロット作成機能
    └── gating.py            # ゲーティング機能
```

## 📖 使用方法

### 1. ファイルアップロード
- サイドバーからFCSファイルを選択・アップロード
- ファイル情報と基本統計の確認

### 2. 基本解析
- チャンネル選択とデータプレビュー
- ヒストグラムと散布図の作成
- 統計情報の確認

### 3. 高度な解析
- ゲーティング領域の設定
- 等高線プロットでの密度可視化
- ゲート統計の計算

### 4. 結果のエクスポート
- 解析結果のCSV出力
- プロット画像の保存

## ⚙️ カスタマイズ

### UI設定
- 色、サイズ、レイアウトの調整
- プロット設定: ビン数、カラーマップ、透明度
- ゲーティング設定: ゲート色、線幅、最小ポイント数

### 設定ファイル
`.streamlit/config.toml`でアプリケーションの動作を調整：
- アップロードサイズ制限
- テーマカラー
- パフォーマンス設定

## 🔧 開発・拡張

### 新しいページの追加
`pages/`ディレクトリに新しいPythonファイルを作成し、`app.py`でページを登録

### 新しいプロット機能
`utils/plotting.py`に新しい関数を追加。必要に応じてBokehやPlotlyのウィジェットを実装

### ゲーティング機能の拡張
`utils/gating.py`に新しいゲートタイプを追加し、JavaScriptコールバックを実装

### デバッグ
開発時のデバッグには以下を活用：
```python
import streamlit as st
st.write("Debug info:", variable)  # デバッグ出力
st.json(data)                      # JSONデータの表示
```

## 📋 対応ファイル形式

- **FCS 2.0/3.0/3.1**: 標準的なフローサイトメトリーファイル
- **チャンネル**: FSC, SSC, 各種蛍光チャンネル
- **パラメータ**: リニア・ログスケール対応

## 🔧 トラブルシューティング

### ファイルアップロードエラー
- ファイルサイズ制限（100MB）を確認
- FCS形式の妥当性をチェック

### プロット表示問題
- ブラウザのJavaScript有効化を確認
- キャッシュクリア

### パフォーマンス問題
- データサイズの縮小
- プロットのビン数調整

## 📄 ライセンス

このプロジェクト是研究・教育目的で作成されています。商用利用については開発者にお問い合わせください。

## 🤝 コントリビューション

バグ報告や機能要望は、GitHubのIssuesでお知らせください。プルリクエストも歓迎します。

## 📞 サポート

技術的な質問やサポートが必要な場合は、プロジェクトのメンテナーにお問い合わせください。

---

**Author**: ymmtshch  
**Repository**: [https://github.com/ymmtshch/FACSDataAnalysis](https://github.com/ymmtshch/FACSDataAnalysis)
