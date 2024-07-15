# The CLI entrypoint to vLLM.
import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


def registrer_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def serve(args: argparse.Namespace) -> None:
    # EngineArgs expects the model name to be passed as --model.
    args.model = args.model_tag

    run_server(args)


def interactive_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    if args.command == "complete":
        complete(model_name, openai_client)
    elif args.command == "chat":
        chat(args.system_prompt, model_name, openai_client)


def tgis_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    if args.command == "download-weights":
        download_weights(args.model_name, args.revision, args.token,
                         args.extension, args.auto_convert)
    elif args.command == "convert-to-safetensors":
        convert_to_safetensors(args.model_name, args.revision)
    elif args.command == "convert-to-fast-tokenizer":
        convert_to_fast_tokenizer(args.model_name, args.revision,
                                  args.output_path)


def complete(model_name: str, client: OpenAI) -> None:
    print("Please enter prompt to complete:")
    while True:
        input_prompt = input("> ")

        completion = client.completions.create(model=model_name,
                                               prompt=input_prompt)
        output = completion.choices[0].text
        print(output)


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        input_message = input("> ")
        message = {"role": "user", "content": input_message}
        conversation.append(message)

        chat_completion = client.chat.completions.create(model=model_name,
                                                         messages=conversation)

        response_message = chat_completion.choices[0].message
        output = response_message.content

        conversation.append(response_message)
        print(output)


def download_weights(
    model_name: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
) -> None:
    from vllm.tgis_utils import hub

    print(extension)
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
                print(".safetensors weights not found, \
                    downloading pytorch weights to convert...")
                hub.download_weights(model_name,
                                     ".bin",
                                     revision=revision,
                                     auth_token=token)

            print(".safetensors weights not found, \
                    converting from pytorch weights...")
            convert_to_safetensors(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            print(".safetensors weights not found on hub, \
                    but were found locally. Remove them first to re-convert")
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)


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
        print("Existing .safetensors weights found, \
                remove them first to reconvert")
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
        st_prefix = local_pt_index_file.stem.removeprefix(
            "pytorch_").removesuffix(".bin.index")
        local_st_index_file = (local_pt_index_file.parent /
                               f"{st_prefix}.safetensors.index.json")

        if os.path.exists(local_st_index_file):
            print("Existing .safetensors.index.json file found, \
                    remove it first to reconvert")
            return

        hub.convert_index_file(local_pt_index_file, local_st_index_file,
                               local_pt_files, local_st_files)

    # Convert pytorch weights to safetensors
    hub.convert_files(local_pt_files, local_st_files, discard_names)


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


def _add_query_options(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-Compatible RESTful API server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("The model name used in prompt completion, default to "
              "the first model in list models API call."))
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
        ))
    return parser


def main():
    parser = FlexibleArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI Compatible API server",
        usage="vllm serve <model_tag> [options]")
    serve_parser.add_argument("model_tag",
                              type=str,
                              help="The model tag to serve")
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.set_defaults(dispatch_function=serve)

    complete_parser = subparsers.add_parser(
        "complete",
        help=("Generate text completions based on the given prompt "
              "via the running API server"),
        usage="vllm complete [options]")
    _add_query_options(complete_parser)
    complete_parser.set_defaults(dispatch_function=interactive_cli,
                                 command="complete")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="vllm chat [options]")
    _add_query_options(chat_parser)
    chat_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("The system prompt to be added to the chat template, "
              "used for models that support system prompts."))
    chat_parser.set_defaults(dispatch_function=interactive_cli, command="chat")

    download_weights_parser = subparsers.add_parser(
        "download-weights",
        help=("Download the weights of a given model"),
        usage="vllm download-weights <model_name> [options]")
    download_weights_parser.add_argument("model_name")
    download_weights_parser.add_argument("--revision")
    download_weights_parser.add_argument("--token")
    download_weights_parser.add_argument("--extension", default=".safetensors")
    download_weights_parser.add_argument("--auto_convert", default=True)
    download_weights_parser.set_defaults(dispatch_function=tgis_cli,
                                         command="download-weights")

    convert_to_safetensors_parser = subparsers.add_parser(
        "convert-to-safetensors",
        help=("Convert model weights to safetensors"),
        usage="vllm convert-to-safetensors <model_name> [options]")
    convert_to_safetensors_parser.add_argument("model_name")
    convert_to_safetensors_parser.add_argument("--revision")
    convert_to_safetensors_parser.set_defaults(
        dispatch_function=tgis_cli, command="convert-to-safetensors")

    convert_to_fast_tokenizer_parser = subparsers.add_parser(
        "convert-to-fast-tokenizer",
        help=("Convert to fast tokenizer"),
        usage="vllm convert-to-fast-tokenizer <model_name> [options]")
    convert_to_fast_tokenizer_parser.add_argument("model_name")
    convert_to_fast_tokenizer_parser.add_argument("--revision")
    convert_to_fast_tokenizer_parser.add_argument("--output_path")
    convert_to_fast_tokenizer_parser.set_defaults(
        dispatch_function=tgis_cli, command="convert-to-fast-tokenizer")

    args = parser.parse_args()
    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
