#!/bin/bash

pip install -e .

base_port=$1
python ./debug/offline_parameter_generation.py&
python ./backend/exec.py --port $((base_port + 1))&
python ./backend/exec.py --port $((base_port + 2))&
python ./backend/exec.py --port $((base_port + 3))&
wait