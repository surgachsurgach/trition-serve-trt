#!/bin/bash

source ./bin/common.sh


function usage() {
  echo "Usage: $0 [-t|--target] [-h|--help]"
  echo "  -t|--target: Target to run (data, model, inference)"
  echo "  -h|--help: Print this help message"
  exit 1
}


while [[ $# -gt 0 ]]; do
  case "$1" in
      -t|--target)
        TARGET="${2:-}"
        shift 2 # past argument & value
      ;;
      -h|--help)
        usage
      ;;
      *)    # unknown option
        echo "Unknown option: $1"
        usage
      ;;
  esac
done

print_msg "TARGET: ${TARGET}"
print_msg "deploy code to s3."
TMP_DIR=$(mktemp -d)

mkdir -p ${TMP_DIR}/data/

cp -R ${PROJECT_DIR} ${TMP_DIR}/data/ml
cp -R ${PYLIB_DIR} ${TMP_DIR}/data/pylib


if [ ${TARGET} == "data" ]; then
  cd ${TMP_DIR} && zip -r src.zip data -x \*__pycache__\* -x \*test\* -x *_test.py -x \*test_data\* -x _*.py

  aws s3 sync ${TMP_DIR}/data ${SOURCE_DIR}/data \
    --exclude "**/__pycache__/*" --exclude "**/*test*/*" --exclude "**/_*.py" --exclude "**/.env*" --exclude "**/*.txt" \
    --exclude "**/*.pyc" --exclude "**/Makefile" --exclude "**/*.sh" --exclude "**/*.md" --exclude "**/docker/*"

  aws s3 mv ${TMP_DIR}/src.zip ${SOURCE_DIR}/src.zip

elif [ ${TARGET} == "model" ]; then
  cd ${TMP_DIR} && tar --exclude='__pycache__' --exclude='*test*' --exclude='*_test.py' -czvf code.tar.gz .
  aws s3 mv ${TMP_DIR}/code.tar.gz ${SOURCE_DIR}/

elif [ ${TARGET} == "inference" ]; then
  source .env.inference

  cp ${WORKSPACE_DIR}/${HANDLER_PATH} ${TMP_DIR}/inference.py
  cd ${TMP_DIR} && tar --exclude='__pycache__' --exclude='*test*' --exclude='*_test.py' -czvf code.tar.gz .
  aws s3 mv ${TMP_DIR}/code.tar.gz ${SOURCE_DIR}/

else
  print_error "Invalid target"
  exit 1
fi

cd ${PROJECT_DIR}

trap clear EXIT
