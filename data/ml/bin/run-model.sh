#!/bin/bash

source ./bin/common.sh

print_msg "run model."
PYTHONPATH=${WORKSPACE_DIR} python3 _run_model.py
