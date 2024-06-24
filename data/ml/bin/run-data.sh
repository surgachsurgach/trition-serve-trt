#!/bin/bash

source ./bin/common.sh

print_msg "run data."
PYTHONPATH=${WORKSPACE_DIR} python3 _run_data.py
