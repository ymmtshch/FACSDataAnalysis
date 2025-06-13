# FCS イベントデータ抽出 & CSV ダウンロードアプリ

FlowCytometry Standard（FCS）ファイルからイベントデータを抽出し、CSV形式でダウンロードできるStreamlitアプリケーションです。

## 機能

- FCSファイルのアップロード
- イベントデータの表示（上位10行のプレビュー）
- イベントデータのCSV形式でのダウンロード

## 必要な環境

- Python 3.7以上
- 以下のPythonパッケージ:
  - streamlit
  - flowkit
  - pandas

## インストール

1. リポジトリをクローンします：
```bash
git clone <repository-url>
cd <repository-name>
```

2. 必要なパッケージをインストールします：
```bash
pip install streamlit flowkit pandas
```

または、requirements.txtがある場合：
```bash
pip install -r requirements.txt
```

## 使用方法

1. アプリケーションを起動します：
```bash
streamlit run main31.py
```

2. ブラウザでアプリケーションが開きます（通常は http://localhost:8501）

3. 「FCS ファイルをアップロードしてください」ボタンをクリックして、FCSファイルを選択します

4. ファイルが正常に読み込まれると、イベントデータの上位10行が表示されます

5. 「CSV をダウンロード」ボタンをクリックして、全イベントデータをCSV形式でダウンロードできます

## サポートされるファイル形式

- .fcs（Flow Cytometry Standard）

## 出力

- `events_output.csv`: 抽出されたイベントデータを含むCSVファイル

## 技術的詳細

このアプリケーションは以下のライブラリを使用しています：

- **Streamlit**: Webアプリケーションフレームワーク
- **FlowKit**: FCSファイルの読み込みと処理
- **Pandas**: データフレーム操作とCSV出力

## エラーハンドリング

アプリケーションには基本的なエラーハンドリングが実装されており、FCSファイルの読み込みに失敗した場合はエラーメッセージが表示されます。

## 注意事項

- アップロードされたFCSファイルは一時的に`temp.fcs`として保存されます
- 大きなFCSファイルの場合、処理に時間がかかる場合があります
- イベントデータは生データ（raw）として抽出されます

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細はLICENSEファイルをご覧ください。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## サポート

問題が発生した場合は、Issueを作成してください。
