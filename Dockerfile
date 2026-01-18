FROM python:3.10-slim

# 1. uvの公式イメージからバイナリのみをコピー（最速の導入方法）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. システムパッケージのインストール
# (uvはバイナリの解決能力が高いため、もしすべてのライブラリでwheelが見つかれば
#  これらのaptパッケージの一部は将来的に削除できる可能性があります)
RUN apt-get update && apt-get install -y \
    git \
    libxml2-dev \
    libxslt1-dev \
    build-essential \
    python3-dev \
    && apt-get clean

# 3. 【重要】先にrequirements.txtのみをコピーしてインストール
# プログラムコードの変更でキャッシュが切れないように順番を変えました
COPY requirements.txt /app/

# 4. uvを使ってインストール
# --system: DockerのPython環境に直接インストール
# lxmlなどのビルドオプションもuvが自動解決するため、通常は単純な記述で動作します
RUN uv pip install --system wheel lxml -r requirements.txt

# 5. 最後にソースコードをコピー
# これにより、プログラムを変更しても上記までの手順はキャッシュ(スキップ)されます
COPY ./program /app/program
# COPY ./data /app/data

CMD ["sleep", "infinity"]