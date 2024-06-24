import argparse
import http
import json

import boto3
from loguru import logger

from data.ml import _run_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, required=True)

    args, _ = parser.parse_known_args()
    virtual_cluster_id = _run_data.EksVirtualCluster.PROD_DATA

    emr_client = boto3.client("emr-containers", region_name="us-east-1")
    response = emr_client.cancel_job_run(
        id=args.job_id,
        virtualClusterId=virtual_cluster_id,
    )

    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    if status_code != http.HTTPStatus.OK:
        raise RuntimeError(f"Failed to cancel job: {json.dumps(response, indent=4)}")

    logger.info(f"Status Code: {status_code}")
    logger.info(f"Job Run ID: {response['id']}")
    logger.info(f"Response detail: {json.dumps(response, indent=4)}")


if __name__ == "__main__":
    main()
