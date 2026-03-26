#!/bin/bash

# Run all 9 baseline experiments simultaneously.
# Each gets a unique port and writes to its own log file.
# Usage: bash run_all_baselines.sh

PYTHON=/home/jyoti/miniconda3/bin/python
SCRIPT=src/main.py

echo "Starting all baseline runs..."

BASELINE_MODE=minload_1 RECEIVER_PORT=6001 nohup $PYTHON $SCRIPT >> log_1.txt 2>&1 &
echo "Started minload_1  (port 6001, pid $!)"

BASELINE_MODE=minload_2 RECEIVER_PORT=6002 nohup $PYTHON $SCRIPT >> log_2.txt 2>&1 &
echo "Started minload_2  (port 6002, pid $!)"

BASELINE_MODE=minload_3 RECEIVER_PORT=6003 nohup $PYTHON $SCRIPT >> log_3.txt 2>&1 &
echo "Started minload_3  (port 6003, pid $!)"

BASELINE_MODE=minprop_1 RECEIVER_PORT=6004 nohup $PYTHON $SCRIPT >> log_4.txt 2>&1 &
echo "Started minprop_1  (port 6004, pid $!)"

BASELINE_MODE=minprop_2 RECEIVER_PORT=6005 nohup $PYTHON $SCRIPT >> log_5.txt 2>&1 &
echo "Started minprop_2  (port 6005, pid $!)"

BASELINE_MODE=minprop_3 RECEIVER_PORT=6006 nohup $PYTHON $SCRIPT >> log_6.txt 2>&1 &
echo "Started minprop_3  (port 6006, pid $!)"

BASELINE_MODE=rand_1    RECEIVER_PORT=6007 nohup $PYTHON $SCRIPT >> log_7.txt 2>&1 &
echo "Started rand_1     (port 6007, pid $!)"

BASELINE_MODE=rand_2    RECEIVER_PORT=6008 nohup $PYTHON $SCRIPT >> log_8.txt 2>&1 &
echo "Started rand_2     (port 6008, pid $!)"

BASELINE_MODE=rand_3    RECEIVER_PORT=6009 nohup $PYTHON $SCRIPT >> log_9.txt 2>&1 &
echo "Started rand_3     (port 6009, pid $!)"

echo ""
echo "All 9 runs launched. Monitor with: tail -f log_1.txt"
echo "Check all PIDs with: jobs -l"
