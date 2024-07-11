"""Triton server client for BERT4Rec model."""

import numpy as np
import tritonclient.http as httpclient


def main():
    model_name = "bert4rec_ensemble"
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

    input_inputs = httpclient.InferInput("inputs", [39], "INT64")
    input_inputs.set_data_from_numpy(
        input_tensor=np.array(
            [
                    1,
                    2,
                    1,
                    3,
                    4,
                    5,
                    10,
                    6,
                    11,
                    7,
                    8,
                    9,
                    12,
                    1,
                    2,
                    5,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
            ],
            dtype=np.int64,
        ),
        binary_data=True,
    )


    inputs = [
        input_inputs,
    ]
    outputs = [
        httpclient.InferRequestedOutput("output__0"),
        httpclient.InferRequestedOutput("output__1")
    ]
    response = client.infer(model_name, inputs, outputs=outputs)
    print(response.as_numpy("output__0"))
    print(response.as_numpy("output__1"))


if __name__ == "__main__":
    main()
