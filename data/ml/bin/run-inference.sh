#!/bin/bash

source ./bin/common.sh

print_msg "run inference."
PYTHONPATH=${WORKSPACE_DIR} python3 _run_inference.py
