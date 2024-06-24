import hashlib
import json
import random

import boto3
from botocore import exceptions as botocore_exceptions
import moto
import pytest

from data.pylib import aws_utils


def _generate_random_sha():
    random_sha = hashlib.sha256(f"{random.randint(0, 100)}".encode("utf-8")).hexdigest()
    return f"sha256:{random_sha}"


def _create_image_layers(n):
    layers = []
    for _ in range(n):
        layers.append(
            {
                "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                "size": random.randint(100, 1000),
                "digest": _generate_random_sha(),
            }
        )
    return layers


def _create_image_digest(layers):
    layer_digests = "".join([layer["digest"] for layer in layers])
    summed_digest = hashlib.sha256(f"{layer_digests}".encode("utf-8")).hexdigest()
    return f"sha256:{summed_digest}"


def _create_image_manifest(image_digest=None):
    layers = _create_image_layers(5)
    if image_digest is None:
        image_digest = _create_image_digest(layers)
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "size": sum(layer["size"] for layer in layers),
            "digest": image_digest,
        },
        "layers": layers,
    }


@moto.mock_sts
def test_establish_session():
    try:
        aws_utils.establish_session()
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"establish_session() raised an exception: {e}")


def test_establish_session_retry(mocker):
    mocker.patch("subprocess.run")

    mock_boto3_session = mocker.patch("boto3.Session")
    mock_boto3_session.return_value.client.return_value.get_caller_identity.side_effect = [
        botocore_exceptions.UnauthorizedSSOTokenError(),
        {
            "UserId": "AKIAIOSFODNN7EXAMPLE",
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:user/moto",
        },
    ]

    spy_establish_session = mocker.spy(aws_utils, "establish_session")

    try:
        aws_utils.establish_session()
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"establish_session() raised an exception: {e}")
    assert mock_boto3_session.return_value.client.return_value.get_caller_identity.call_count == 2
    assert spy_establish_session.call_count == 2


def test_establish_session_failed(mocker):
    mocker.patch("subprocess.run")

    mock_boto3_session = mocker.patch("boto3.Session")
    mock_boto3_session.return_value.client.return_value.get_caller_identity.side_effect = botocore_exceptions.UnauthorizedSSOTokenError()

    with pytest.raises(RuntimeError):
        aws_utils.establish_session()


def test_get_account_id_and_region_from_ecr_registry():
    assert aws_utils.get_account_id_and_region_from_ecr_registry("123456789012.dkr.ecr.ap-northeast-2.amazonaws.com") == (
        "123456789012",
        "ap-northeast-2",
    )


@moto.mock_ecr
def test_delete_tag_from_ecr_repository():
    repo_name = "test-repository"
    ecr = boto3.client("ecr", region_name="ap-northeast-2")
    ecr.create_repository(repositoryName=repo_name)
    ecr.put_image(
        repositoryName=repo_name,
        imageManifest=json.dumps(_create_image_manifest()),
        imageTag="test-tag",
    )

    aws_utils.delete_tag_from_ecr_repository("test-repository", "test-tag")
    assert ecr.describe_images(repositoryName="test-repository")["imageDetails"] == []


def test_get_account_name():
    assert aws_utils.Account.get_account_name("995621642422") == "stage"
    assert aws_utils.Account.get_account_name("697122891294") == "prod"
    assert aws_utils.Account.get_account_name("368316345532") == "prod_data"
    assert aws_utils.Account.get_account_name("801714584815") == "infra"
