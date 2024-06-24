ARG BASE_IMAGE=801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:emr-6.10.0-python-3.10.9
FROM $BASE_IMAGE

USER root

COPY ml/requirements.data.txt /usr/requirements.txt

RUN pip3 install --no-cache-dir -r /usr/requirements.txt

USER hadoop:hadoop
