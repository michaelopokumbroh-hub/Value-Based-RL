#!/bin/bash
# Spring 2026, Deep Learning - Lab 5
# Evaluation script for Task 3: Enhanced DQN on ALE/Pong-v5
# Student ID: 114202103
#
# HOW TO RUN (Windows Anaconda Prompt):
#   cd LAB5_114202103_Code
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_600000.pt
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_1000000.pt
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_1500000.pt
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_2000000.pt
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_2500000.pt
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task3.py --model_path ..\LAB5_114202103_task3_best.pt


export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== Task 3 | 600,000 steps ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_600000.pt

echo "=== Task 3 | 1,000,000 steps ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_1000000.pt

echo "=== Task 3 | 1,500,000 steps ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_1500000.pt

echo "=== Task 3 | 2,000,000 steps ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_2000000.pt

echo "=== Task 3 | 2,500,000 steps ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_2500000.pt

echo "=== Task 3 | Best model ==="
python test_model_task3.py --model_path ../LAB5_114202103_task3_best.pt
