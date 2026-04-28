#!/bin/bash

# Run all 9 baseline experiments simultaneously.
# Each gets a unique port, its own training_logs folder, and its own log file.
# CSVs are saved to src/results/
# Usage: bash run_all_baselines.sh

PYTHON=/home/jyoti/miniconda3/bin/python
SCRIPT=src/main.py

echo "Starting all baseline runs..."

BASELINE_MODE=minload_1 RECEIVER_PORT=6001 TRAINING_LOG_FOLDER=training_logs_minload_1 nohup $PYTHON $SCRIPT >> log_1.txt 2>&1 &
echo "Started minload_1  (port 6001, logs -> training_logs_minload_1, pid $!)"

BASELINE_MODE=minload_2 RECEIVER_PORT=6002 TRAINING_LOG_FOLDER=training_logs_minload_2 nohup $PYTHON $SCRIPT >> log_2.txt 2>&1 &
echo "Started minload_2  (port 6002, logs -> training_logs_minload_2, pid $!)"

BASELINE_MODE=minload_3 RECEIVER_PORT=6003 TRAINING_LOG_FOLDER=training_logs_minload_3 nohup $PYTHON $SCRIPT >> log_3.txt 2>&1 &
echo "Started minload_3  (port 6003, logs -> training_logs_minload_3, pid $!)"

BASELINE_MODE=minprop_1 RECEIVER_PORT=6004 TRAINING_LOG_FOLDER=training_logs_minprop_1 nohup $PYTHON $SCRIPT >> log_4.txt 2>&1 &
echo "Started minprop_1  (port 6004, logs -> training_logs_minprop_1, pid $!)"

BASELINE_MODE=minprop_2 RECEIVER_PORT=6005 TRAINING_LOG_FOLDER=training_logs_minprop_2 nohup $PYTHON $SCRIPT >> log_5.txt 2>&1 &
echo "Started minprop_2  (port 6005, logs -> training_logs_minprop_2, pid $!)"

BASELINE_MODE=minprop_3 RECEIVER_PORT=6006 TRAINING_LOG_FOLDER=training_logs_minprop_3 nohup $PYTHON $SCRIPT >> log_6.txt 2>&1 &
echo "Started minprop_3  (port 6006, logs -> training_logs_minprop_3, pid $!)"

BASELINE_MODE=rand_1    RECEIVER_PORT=6007 TRAINING_LOG_FOLDER=training_logs_rand_1    nohup $PYTHON $SCRIPT >> log_7.txt 2>&1 &
echo "Started rand_1     (port 6007, logs -> training_logs_rand_1,    pid $!)"

BASELINE_MODE=rand_2    RECEIVER_PORT=6008 TRAINING_LOG_FOLDER=training_logs_rand_2    nohup $PYTHON $SCRIPT >> log_8.txt 2>&1 &
echo "Started rand_2     (port 6008, logs -> training_logs_rand_2,    pid $!)"

BASELINE_MODE=rand_3    RECEIVER_PORT=6009 TRAINING_LOG_FOLDER=training_logs_rand_3    nohup $PYTHON $SCRIPT >> log_9.txt 2>&1 &
echo "Started rand_3     (port 6009, logs -> training_logs_rand_3,    pid $!)"

echo ""
echo "All 9 runs launched."
echo "CSVs will appear in: src/logs/"
echo "Monitor a run with:  tail -f log_1.txt"
echo "Check all PIDs with: ps aux | grep main.py"
