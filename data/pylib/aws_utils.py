from typing import Optional

import boto3
from botocore import exceptions as botocore_exceptions
from loguru import logger

from data.pylib import command_utils
from data.pylib import strenum


class Account(strenum.StrEnum):
    """AWS account enum."""

    dev = "119269236144"
    emr_public = "996579266876"
    deeplearning_public = "763104351884"
    infra = "801714584815"
    intranet = "927952490827"
    prod = "697122891294"
    prod_data = "368316345532"
    qa = "501660728113"
    sandbox = "268847686663"
    stage = "995621642422"
    stage_data = "171943046871"

    @staticmethod
    def get_account_name(account_id: str) -> str:
        """Returns account name by account ID.

        Args:
            account_id: AWS account ID.

        Returns:
            Account name.
        """
        return Account(account_id).name


def establish_session(profile_name: Optional[str] = None, region_name: str = "ap-northeast-2", retry: bool = True):
    """Establishes AWS session.

    Establishes AWS session by using boto3.Session or boto3.client.
    If profile_name is provided, boto3.Session is used, otherwise boto3.client is used.

    Args:
        profile_name: AWS profile name. If not provided, default profile is used.
        region_name: AWS region name. Default is ap-northeast-2.
        retry: If True, retry to establish session if failed to login to AWS SSO.

    Raises:
        RuntimeError: If failed to login to AWS SSO.
    """

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sts = session.client("sts")

    try:
        identity = sts.get_caller_identity()
        logger.info(f"Established AWS session for {identity['Account']} ({identity['Arn']})")
    except botocore_exceptions.UnauthorizedSSOTokenError as e:
        if retry:
            command = ["aws", "sso", "login"]
            if profile_name:
                command.extend(["--profile", profile_name])
            command_utils.run(command)
            return establish_session(profile_name, retry=False)
        raise RuntimeError("Failed to login to AWS SSO") from e


def login_to_ecr(account_id: str, region_name: str, profile_name: Optional[str] = None, retry: bool = True):
    """Login to AWS ECR.

    Login to AWS ECR by using aws ecr get-login-password and docker login.

    Args:
        account_id: AWS account ID.
        region_name: AWS region name.
        profile_name: AWS profile name.
        retry: If True, retry to login to AWS ECR if failed to login to AWS SSO.

    Raises:
        RuntimeError: If failed to login to AWS ECR.
    """

    command = "aws ecr get-login-password "
    if profile_name:
        command += f"--profile {profile_name} "
    command += f"--region {region_name} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region_name}.amazonaws.com"

    result = command_utils.run(command, ignore_error=True)
    # If failed to login to ECR, retry to login to AWS SSO and login to ECR again.
    if not result:
        if retry:
            establish_session(profile_name=profile_name, region_name=region_name)
            login_to_ecr(account_id, region_name, profile_name=profile_name, retry=False)
        else:
            raise RuntimeError("Failed to login to AWS ECR")


def get_account_id_and_region_from_ecr_registry(ecr_registry: str):
    """Extracts AWS account ID and region name from AWS ECR registry.

    Args:
        ecr_registry: AWS ECR registry.

    Returns:
        AWS account ID and region name.
        For example, if ecr_registry is 119269236144.dkr.ecr.ap-northeast-2.amazonaws.com,
        returns (119269236144, ap-northeast-2).
    """
    account_id, region_name = ecr_registry.split(".")[0], ecr_registry.split(".")[3]
    return account_id, region_name


def delete_tag_from_ecr_repository(
    repository_name: str, tag: str, profile_name: Optional[str] = None, region_name: Optional[str] = "ap-northeast-2"
):
    """Deletes tag from AWS ECR repository.

    Args:
        repository_name: AWS ECR repository name.
        tag: AWS ECR tag.
        region_name: AWS region name. Default is ap-northeast-2.
        profile_name: AWS profile name. If not provided, default profile is used.
    """

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    ecr = session.client("ecr")

    ecr.batch_delete_image(repositoryName=repository_name, imageIds=[{"imageTag": tag}])


def get_account_name(account_id: str):
    """Returns AWS account name.

    Args:
        account_id: AWS account ID.
    """

    return Account(account_id).name
