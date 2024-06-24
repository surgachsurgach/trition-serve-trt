#!/usr/bin/env bash

source ./bin/common.sh
source .env  # load environment variables

BUILD_CONTEXT=$WORKSPACE_DIR/data
REGISTRY=801714584815.dkr.ecr.${AWS_REGION}.amazonaws.com
REPOSITORY=data-ml-recsys


TMP_DIR=$(mktemp -d)

print_msg "parse arguments."
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --t|--target)
      TARGET="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      echo "Usage: deploy_image.sh -t <TARGET>"
      echo "  -t, --target. choices: data, model"
      exit 0
      ;;
    *)    # unknown option
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

print_msg "target is $TARGET"
print_msg "select the Dockerfile."
if [ "$TARGET" == "data" ]; then
  DOCKERFILE="data/ml/docker/data/Dockerfile"
  TAG=$REGISTRY/$REPOSITORY:data-$USER
  PLATFORM="linux/amd64,linux/arm64"
elif [ "$TARGET" == "model" ]; then
  DOCKERFILE="data/ml/docker/model/Dockerfile"
  TAG=$REGISTRY/$REPOSITORY:model-$USER
  PLATFORM="linux/amd64"
elif [ "$TARGET" == "inference" ]; then
  DOCKERFILE="data/ml/docker/inference/Dockerfile"
  TAG=$REGISTRY/$REPOSITORY:inference-$USER
  PLATFORM="linux/amd64"
else
  print_error "Invalid target"
  exit 1
fi

print_msg "login to ECR."

aws ecr get-login-password --profile "${AWS_INFRA_PROFILE}" --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}" \
  || { echo "Failed to login to ECR. Please check AWS_INFRA_PROFILE, AWS_REGION is set correctly."; exit 1; }

print_msg "build and push the image"

print_msg "Delete image tag $TAG if exists."
aws ecr batch-delete-image \
  --profile "${AWS_INFRA_PROFILE}" \
  --repository-name "${REPOSITORY}" \
  --region "${AWS_REGION}" \
  --image-ids imageTag="${TAG}" > /dev/null

cd $WORKSPACE_DIR || exit 1

if [ "$TARGET" == "data" ]; then
  docker buildx build\
    --platform="${PLATFORM}" \
    -f "${DOCKERFILE}" \
    -t "${TAG}" --push \
    "${BUILD_CONTEXT}"

elif [ "$TARGET" == "model" ]; then
  docker buildx inspect ml-builder || docker buildx create --name ml-builder --use
  docker buildx build \
    --provenance=false \
    --platform="${PLATFORM}" \
    --target sagemaker \
    -f "${DOCKERFILE}" \
    -t "${TAG}" --push \
    "${BUILD_CONTEXT}"
elif [ "$TARGET" == "inference" ]; then
  docker buildx inspect ml-builder || docker buildx create --name ml-builder --use
  docker buildx build \
    --provenance=false \
    --platform="${PLATFORM}" \
    -f "${DOCKERFILE}" \
    -t "${TAG}" --push \
    "${BUILD_CONTEXT}"
fi


if [ $? -eq 0 ]; then
  print_msg "Successfully built and pushed the image."
else
  print_error "Failed to build and push the image."
  print_msg "Please check the error message and try again."
  exit 1
fi

trap clear EXIT
