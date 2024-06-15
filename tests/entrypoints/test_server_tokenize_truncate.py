import grpc
# imports for guided decoding tests
import pytest
# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray

from ..utils import ServerRunner
# to install pb, run Makefile to compile grpc protobuf
from .pb import generation_pb2 as pb2
from .pb import generation_pb2_grpc as gpb2

# Config. vars for gRPC
SERVER = 'localhost'
PORT = 8033

# The tokenizer was tested using the following model:
MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def server():
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16"
    ])
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


# Fixture to create a gRPC stub for the GenerationService
@pytest.fixture(scope="module")
def grpc_stub():
    channel = grpc.insecure_channel(f"{SERVER}:{PORT}")
    stub = gpb2.GenerationServiceStub(channel)
    yield stub
    channel.close()


# Test cases
@pytest.mark.parametrize("test_case", [
    {
        "name": "Tokenize with offsets",
        "request": {
            "text": "The very long story is written",
            "return_offsets": True,
        },
        "response": {
            "tokenCount":
            7,
            "offsets": [
                {
                    "start": 0,
                    "end": 0
                },
                {
                    "start": 0,
                    "end": 3
                },
                {
                    "start": 3,
                    "end": 8
                },
                {
                    "start": 8,
                    "end": 13
                },
                {
                    "start": 13,
                    "end": 19
                },
                {
                    "start": 19,
                    "end": 22
                },
                {
                    "start": 22,
                    "end": 30
                },
            ],
        },
    },
    {
        "name": "Tokenize with tokens and offsets",
        "request": {
            "text": "The very long story is written",
            "return_tokens": True,
            "return_offsets": True,
        },
        "response": {
            "tokenCount":
            7,
            "tokens":
            ["</s>", "The", "Ġvery", "Ġlong", "Ġstory", "Ġis", "Ġwritten"],
            "offsets": [
                {
                    "start": 0,
                    "end": 0
                },
                {
                    "start": 0,
                    "end": 3
                },
                {
                    "start": 3,
                    "end": 8
                },
                {
                    "start": 8,
                    "end": 13
                },
                {
                    "start": 13,
                    "end": 19
                },
                {
                    "start": 19,
                    "end": 22
                },
                {
                    "start": 22,
                    "end": 30
                },
            ],
        },
    },
    {
        "name": "Tokenize with tokens and truncation",
        "request": {
            "text": "The very long story is written by a very long story",
            "return_tokens": True,
            "truncate_input_tokens": 10,
        },
        "response": {
            "tokenCount":
            10,
            "tokens": [
                "Ġvery",
                "Ġlong",
                "Ġstory",
                "Ġis",
                "Ġwritten",
                "Ġby",
                "Ġa",
                "Ġvery",
                "Ġlong",
                "Ġstory",
            ],
        },
    },
    {
        "name":
        "Tokenize, trunc and offset for a request with no text message",
        "request": {
            "text": "",
            "return_offsets": True,
            "return_tokens": True,
            "truncate_input_tokens": 10,
        },
        "response": {
            "tokenCount": 1,
            "tokens": ["</s>"],
        },
    },
    {
        "name": "A request without text ('') and parameters",
        "request": {
            "text": ""
        },
        "response": {
            "tokenCount": 1
        },
    },
    {
        "name": "A request without text (None) and parameters",
        "request": {
            "text": None
        },
        "response": {
            "tokenCount": 1
        },
    },
])
def test_tokenization(server, grpc_stub, test_case):
    """Test tokenization with the given test case."""
    request = test_case['request']
    text = request['text']
    truncate_input_tokens = request.get('truncate_input_tokens', None)

    # Construct the request
    batch = pb2.BatchedTokenizeRequest(
        model_id="unused",
        requests=[pb2.TokenizeRequest(text=text)],
        return_tokens=request.get('return_tokens', False),
        return_offsets=request.get('return_offsets', False),
        truncate_input_tokens=truncate_input_tokens)

    try:
        responses = grpc_stub.Tokenize(batch).responses
    except grpc.RpcError as e:
        # Print debug message in case of connection failure
        print(f"Failed to connect to the gRPC server: {e}")
        pytest.fail(f"gRPC call failed with error: {e}")

    # Verify the response
    expected_response = test_case['response']
    resp = responses[0]

    assert resp.token_count == expected_response['tokenCount'],\
        "Token count mismatch"
    if 'tokens' in expected_response:
        assert resp.tokens == expected_response['tokens'],\
            "Tokens mismatch"
    if 'offsets' in expected_response:
        expected_offsets = expected_response['offsets']
        assert len(resp.offsets) == len(expected_offsets),\
            "Offset length mismatch"
        for resp_offset, exp_offset in zip(resp.offsets, expected_offsets):
            assert resp_offset.start == exp_offset.get('start', None),\
                "Start offset mismatch"
            assert resp_offset.end == exp_offset.get('end', None),\
                "End offset mismatch"

    print("Test case passed: ", test_case["name"])


if __name__ == "__main__":
    pytest.main([__file__])
