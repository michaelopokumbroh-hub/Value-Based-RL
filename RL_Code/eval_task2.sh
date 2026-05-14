#!/bin/bash
# Spring 2026, Deep Learning - Lab 5
# Evaluation script for Task 2: Vanilla DQN on ALE/Pong-v5
# Student ID: 114202103
#
# HOW TO RUN (Windows Anaconda Prompt):
#   cd LAB5_114202103_Code
#   set KMP_DUPLICATE_LIB_OK=TRUE && python test_model_task2.py --model-path ..\LAB5_114202103_task2.pt

export KMP_DUPLICATE_LIB_OK=TRUE
python test_model_task2.py --model-path ../LAB5_114202103_task2.pt
