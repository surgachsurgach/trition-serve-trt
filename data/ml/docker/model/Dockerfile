ARG BASE_IMAGE=801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:pytorch-2.2.0-python-3.10
FROM $BASE_IMAGE as base

COPY ml/requirements.model.txt /usr/requirements.txt
RUN pip install --no-cache-dir -r /usr/requirements.txt

ARG BASE_IMAGE=801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:pytorch-2.2.0-python-3.10-sagemaker
FROM $BASE_IMAGE as sagemaker
# This is the image runned by Sagemaker container runtime launch mechanism.
# user code is loaded from s3 and mounted to /opt/ml/code at container runtime, then executed by the sagemaker entrypoint script.

COPY ml/requirements.model.txt /usr/requirements.txt
RUN pip install --no-cache-dir -r /usr/requirements.txt


FROM base as k8s
ARG SOURCE_DIR=/opt/ml/code
# This is image runned by container runtime.
# user code is copied to /opt/ml/code at build time, then executed by the user entrypoint script.

# should be the same as the sagemaker image
WORKDIR $SOURCE_DIR

COPY ml ./data/ml
COPY pylib ./data/pylib

ENV PYTHONPATH=$SOURCE_DIR
ENV GIN_SEARCH_PATH=$SOURCE_DIR

ENTRYPOINT ["python3", "data/ml/model_runner/main.py"]
