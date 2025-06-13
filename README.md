# FACS Data Analysis アプリ

このアプリは、**Flow Cytometry Standard（.fcs）ファイル**からイベントデータを抽出し、CSV形式でダウンロードできる **Streamlit** ベースのWebアプリケーションです。FACS（蛍光抗体細胞ソーティング）の生データを手軽に確認・変換する用途に適しています。

---

## 🚀 デモサイト

▶️ [こちらをクリックしてアプリを試す](https://facsdataanalysis-zz29jykc7nfshywbujaj8q.streamlit.app/)

---

## 🔧 主な機能

- `.fcs` ファイルのアップロード
- イベントデータ（イベント × パラメータ）の抽出と表示（上位10行をプレビュー）
- イベント数・パラメータ数・ファイルサイズの統計表示
- メタデータ（FCSヘッダー情報）の表示オプション
- 抽出されたイベントデータのCSVダウンロード

## 🌐 Streamlit Cloudでの起動方法

1. このリポジトリを **Fork** または **Clone** してください
2. [Streamlit Cloud](https://streamlit.io/cloud) にログインして、「New app」からリポジトリを指定
3. 以下の設定を入力します：
   - **Main file path**: `app.py`
   - **Python version**: `3.9` など（`packages.txt` に記載済みのバージョンに準拠）
   - **Dependencies**: 自動で `requirements.txt`、`packages.txt` が読み込まれます
4. 「Deploy」をクリックすれば完了です

## 💻 ローカル環境での実行方法

### 必要要件

- Python 3.8以上
- pip または conda

### インストール

```bash
pip install -r requirements.txt
```

### 実行

```bash
streamlit run app.py
```

## 🧪 使い方

1. アプリを起動し、`.fcs` ファイルをアップロードします
2. ファイルが正常に読み込まれると、データのプレビューと統計情報が表示されます
3. 必要に応じてメタデータも表示可能です
4. 「CSV をダウンロード」ボタンで、抽出結果をCSV形式で保存できます

## 📂 対応ファイル形式

- `.fcs`（Flow Cytometry Standard ファイル）

## ⚠️ 注意事項

- `.fcs` ファイルの構造やサイズによっては、読み込みに時間がかかる場合があります
- 本アプリでは、ゲーティング処理やログ変換などの前処理は行っていません（生データのまま出力されます）
- メタデータはFCSのヘッダー情報であり、全ての項目がユーザーにとって有用とは限りません

## 📁 ファイル構成

```
├── app.py              # Streamlitアプリ本体
├── README.md           # 本ファイル
├── requirements.txt    # pip依存パッケージ
├── packages.txt        # Pythonバージョン（Streamlit Cloud用）
├── LICENSE.txt         # ライセンス情報
```

## 📜 ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。詳細は `LICENSE.txt` をご確認ください。

## 🙋‍♀️ 開発者

このアプリは [ymmtshch] によって開発されました。  
改善提案・バグ報告・機能追加などの Pull Request は歓迎します。
