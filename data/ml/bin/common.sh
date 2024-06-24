#!/bin/bash

source .env

RED='\033[0;31m'
NOCOLOR='\033[0m'
BOLD_GREEN="\033[1;32m"

print_error() {
  MESSAGE=$1
  echo -e "${RED}${MESSAGE}${NOCOLOR}"
}

print_msg() {
    echo -e "${BOLD_GREEN}$*${NOCOLOR}";
} >&2

clear() {
    rm -rf "${TMP_DIR}"
}

print_msg "Setting up common environment variables"
WORKSPACE_DIR=$(git rev-parse --show-toplevel)
PROJECT_DIR="$WORKSPACE_DIR/data/ml"
PYLIB_DIR="$WORKSPACE_DIR/data/pylib"

if [ "${DATA_ACCOUNT}" = "stage-data" ]; then
  S3_BUCKET="ridi-ml-batch-dev"
elif [ "${DATA_ACCOUNT}" = "prod-data" ]; then
  S3_BUCKET="ridi-ml-batch"
else
  print_error "DATA_ACCOUNT must be one of stage-data, prod-data" && exit 1
fi

SOURCE_DIR="s3://${S3_BUCKET}/env/users/${USER}"

export USER
export AWS_INFRA_PROFILE
export WORKSPACE_DIR
export PROJECT_DIR
export PYLIB_DIR
export SOURCE_DIR
export SLACK_LOGGING_URL
