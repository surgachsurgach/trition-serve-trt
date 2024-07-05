"""Triton server client for BERT4Rec model."""

import numpy as np
import tritonclient.http as httpclient


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

    input_inputs = httpclient.InferInput("inputs", [1, 40], "INT64")
    input_inputs.set_data_from_numpy(
        input_tensor=np.array(
            [
                [
                    0,
                    1,
                    0,
                    2,
                    3,
                    4,
                    10,
                    6,
                    11,
                    7,
                    8,
                    9,
                    12,
                    0,
                    1,
                    4,
                    6247,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                    6246,
                ]
            ],
            dtype=np.int64,
        ),
        binary_data=True,
    )

    input_target_idx = httpclient.InferInput("target_idx", [1, 1], "INT64")
    input_target_idx.set_data_from_numpy(
        input_tensor=np.array([[1]], dtype=np.int64),
        binary_data=True,
    )

    inputs = [
        input_inputs,
        input_target_idx,
    ]
    outputs = [
        httpclient.InferRequestedOutput("output__0"),
    ]
    response = client.infer("bert4rec", inputs, outputs=outputs)
    print(response.as_numpy("output__0"))


if __name__ == "__main__":
    main()
