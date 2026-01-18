# GitHubアップロード実装計画

## ゴール
現在のディレクトリを安全に [GitHubリポジトリ](https://github.com/kuroishirai/tse-replication-package-1-million-fuzzing-sessions.git) にアップロードします。

## ユーザー承認済み事項
- **リポジトリサイズ**: `data/database/` と `data/processed_data/` は `.gitignore` で除外します。
- **機密情報**:
    - `program/preparation/user_corpus.py`: Tokenを削除し、入力式に変更します。
    - `program/envFile.ini`: 再現用のため、そのまま含めます（変更なし）。
- **絶対パス**:
    - `data/processed_data/csv/merged_output.csv` 等に含まれていますが、今回は修正せずそのままにします（データフォルダごと除外されるため問題なし、または将来的に除外されるため）。

## 変更内容

### 設定
#### [新規] [.gitignore](file:///Users/tatsuya-shi/research/FuzzingEffectiveness/.gitignore)
- `data/database/`
- `data/processed_data/`
- `__pycache__/`
- `.DS_Store`

### コード修正
#### [変更] [user_corpus.py](file:///Users/tatsuya-shi/research/FuzzingEffectiveness/program/preparation/user_corpus.py)
- ハードコードされた `GITHUB_TOKEN` を削除。
- `input()` または環境変数 `GITHUB_TOKEN` から読み込むロジックに変更。

## 検証
- `grep` でToken削除を確認。
- `git status` で除外ファイルを確認。
- `git push` でアップロード完了を確認。
