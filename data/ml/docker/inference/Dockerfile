ARG BASE_IMAGE=801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:pytorch-2.0.1-python-3.10.11-inference-sagemaker
FROM $BASE_IMAGE

# Install python dependencies.
COPY ml/requirements.model.txt /usr/requirements.model.txt

RUN pip3 install -r /usr/requirements.model.txt
