#!/bin/bash

# Speculative Decoding 예제 실행 스크립트

echo "=== Affine Alignment based Speculative Decoder ==="
echo ""

# 환경 변수 설정 (GPU 메모리 관련)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python 경로 확인
echo "Python version:"
python --version
echo ""

# 예제 실행
echo "Running basic usage example..."
cd examples
python basic_usage.py

echo ""
echo "Example completed!" 