from pathlib import Path

from vllm.cli import convert_to_fast_tokenizer
from vllm.tgis_utils.hub import (download_weights, get_model_path,
                                 local_weight_files)


def test_convert_to_fast_tokenizer():
    model_name = "EleutherAI/gpt-neo-125m"
    # make sure to include .json to download the
    # tokenizer.json and tokenizer_config.json
    download_weights(model_name, extension=[".safetensors", ".json"])
    model_path = get_model_path(model_name)
    local_json_files = [
        Path(p) for p in local_weight_files(model_path, ".json")
    ]
    tokenizer_file = [
        file for file in local_json_files if file.name == "tokenizer.json"
    ][0]
    assert tokenizer_file is not None

    # remove the tokenizer file
    Path.unlink(tokenizer_file)
    local_files_remove_tokenizer = [
        Path(p) for p in local_weight_files(model_path, ".json")
    ]
    assert "tokenizer.json" not in [
        file.name for file in local_files_remove_tokenizer
    ]

    # this should convert the tokenizer_config.json to tokenizer.json
    convert_to_fast_tokenizer(model_name)
    local_files_with_tokenizer = [
        Path(p) for p in local_weight_files(model_path, ".json")
    ]
    assert "tokenizer.json" in [
        file.name for file in local_files_with_tokenizer
    ]
