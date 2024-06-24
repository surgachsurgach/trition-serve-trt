import torch


def assert_dict_equals(actual: dict, expected: dict):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, expected[key])
        elif isinstance(value, dict):
            assert_dict_equals(value, expected[key])
        else:
            assert value == expected[key]
