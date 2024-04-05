import os
from enum import Enum
from typing import Optional

import typer

from pathlib import Path

app = typer.Typer()


@app.command()
def download_weights(
    model_name: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
):
    from vllm.tgis_utils import hub

    meta_exts = [".json", ".py", ".model", ".md"]

    extensions = extension.split(",")

    if len(extensions) == 1 and extensions[0] not in meta_exts:
        extensions.extend(meta_exts)

    files = hub.download_weights(model_name,
                                 extensions,
                                 revision=revision,
                                 auth_token=token)

    if auto_convert and ".safetensors" in extensions:
        if not hub.local_weight_files(hub.get_model_path(model_name, revision),
                                      ".safetensors"):
            if ".bin" not in extensions:
                print(
                    ".safetensors weights not found, downloading pytorch weights to convert..."
                )
                hub.download_weights(model_name,
                                     ".bin",
                                     revision=revision,
                                     auth_token=token)

            print(
                ".safetensors weights not found, converting from pytorch weights..."
            )
            convert_to_safetensors(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            print(
                ".safetensors weights not found on hub, but were found locally. Remove them first to re-convert"
            )
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)


@app.command()
def convert_to_safetensors(
    model_name: str,
    revision: Optional[str] = None,
):
    from vllm.tgis_utils import hub

    # Get local pytorch file paths
    model_path = hub.get_model_path(model_name, revision)
    local_pt_files = hub.local_weight_files(model_path, ".bin")
    local_pt_index_files = hub.local_index_files(model_path, ".bin")
    if len(local_pt_index_files) > 1:
        print(
            f"Found more than one .bin.index.json file: {local_pt_index_files}"
        )
        return

    if not local_pt_files:
        print("No pytorch .bin files found to convert")
        return

    local_pt_files = [Path(f) for f in local_pt_files]
    local_pt_index_file = local_pt_index_files[
        0] if local_pt_index_files else None

    # Safetensors final filenames
    local_st_files = [
        p.parent / f"{p.stem.removeprefix('pytorch_')}.safetensors"
        for p in local_pt_files
    ]

    if any(os.path.exists(p) for p in local_st_files):
        print(
            "Existing .safetensors weights found, remove them first to reconvert"
        )
        return

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name,
            revision=revision,
        )
        architecture = config.architectures[0]

        class_ = getattr(transformers, architecture)

        # Name for this variable depends on transformers version
        discard_names = getattr(class_, "_tied_weights_keys", [])
        discard_names.extend(
            getattr(class_, "_keys_to_ignore_on_load_missing", []))

    except Exception:
        discard_names = []

    if local_pt_index_file:
        local_pt_index_file = Path(local_pt_index_file)
        st_prefix = local_pt_index_file.stem.removeprefix('pytorch_').rstrip(
            '.bin.index')
        local_st_index_file = local_pt_index_file.parent / f"{st_prefix}.safetensors.index.json"

        if os.path.exists(local_st_index_file):
            print(
                "Existing .safetensors.index.json file found, remove it first to reconvert"
            )
            return

        hub.convert_index_file(local_pt_index_file, local_st_index_file,
                               local_pt_files, local_st_files)

    # Convert pytorch weights to safetensors
    hub.convert_files(local_pt_files, local_st_files, discard_names)


@app.command()
def convert_to_fast_tokenizer(
    model_name: str,
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
):
    from vllm.tgis_utils import hub

    # Check for existing "tokenizer.json"
    model_path = hub.get_model_path(model_name, revision)

    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        print(f"Model {model_name} already has a fast tokenizer")
        return

    if output_path is not None:
        if not os.path.isdir(output_path):
            print(f"Output path {output_path} must exist and be a directory")
            return
    else:
        output_path = model_path

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           revision=revision)
    tokenizer.save_pretrained(output_path)

    print(f"Saved tokenizer to {output_path}")


if __name__ == "__main__":
    app()
