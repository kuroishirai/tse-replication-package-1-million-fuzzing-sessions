#!/bin/bash

# エラーが発生した場合、直ちにスクリプトを終了する
set -e

echo "========================================================"
echo "Starting Reproduction of All Research Questions (RQ1-RQ4)"
echo "========================================================"

# Pythonモジュールのパスを通すための環境変数はスクリプト内でsys.pathに追加されているため、
# ここでは単純にスクリプトを実行するだけでOK

# --- RQ1 ---
echo ""
echo "[1/6] Running RQ1: Detection Rate Analysis..."
echo "Executing: python3 program/research_questions/rq1_detection_rate.py"
python3 program/research_questions/rq1_detection_rate.py

# --- RQ2 ---
echo ""
echo "[2/6] Running RQ2: Coverage and Added Analysis..."
echo "Executing: python3 program/research_questions/rq2_coverage_and_added.py"
python3 program/research_questions/rq2_coverage_and_added.py

echo ""
echo "[3/6] Running RQ2: Coverage Count Analysis..."
echo "Executing: python3 program/research_questions/rq2_coverage_count.py"
python3 program/research_questions/rq2_coverage_count.py

# --- RQ3 ---
echo ""
echo "[4/6] Running RQ3: Diff Coverage at Detection..."
echo "Executing: python3 program/research_questions/rq3_diff_coverage_at_detection.py"
python3 program/research_questions/rq3_diff_coverage_at_detection.py

# --- RQ4 ---
# rq4a_bug.py と rq4b_coverage.py はどちらが先でも良いが、a -> b の順で実行
echo ""
echo "[5/6] Running RQ4a: Bug Analysis..."
echo "Executing: python3 program/research_questions/rq4a_bug.py"
python3 program/research_questions/rq4a_bug.py

echo ""
echo "[6/6] Running RQ4b: Coverage Analysis..."
echo "Executing: python3 program/research_questions/rq4b_coverage.py"
python3 program/research_questions/rq4b_coverage.py

echo ""
echo "========================================================"
echo "All Research Questions have been reproduced successfully!"
echo "Results are saved in the 'data/result_data' directory."
echo "========================================================"
