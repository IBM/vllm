SHELL := /bin/bash

server_image_name := tgis-vllm
server_image_target := vllm-openai

##@ Development Tasks

has_gawk := $(shell gawk --version 2>/dev/null)
.PHONY: help
help: ## Display this help.
ifdef has_gawk
	@gawk -f ./scripts/makefile.help.awk $(MAKEFILE_LIST)
else
	@awk 'BEGIN{FS=":.*##"; printf("\nUsage:\n  make \033[36m<target>\033[0m\n\n")} /^[-a-zA-Z_0-9\\.]+:.*?##/ {t=$$1; if(!(t in p)){p[t]; printf("\033[36m%-15s\033[0m %s\n", t, $$2)}}' $(MAKEFILE_LIST)
	@echo
	@echo "NOTE: Help output with headers requires GNU extensions to awk. Please install gawk for the best experience."
endif

target_path := "vllm/entrypoints/grpc/pb"
gen-protos:
	# Compile protos
	pip install grpcio-tools==1.62.0 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4'
	mkdir -p $(target_path)
	python -m grpc_tools.protoc -Iproto --python_out=$(target_path) \
		--grpc_python_out=$(target_path) --mypy_out=$(target_path) proto/generation.proto
	find $(target_path)/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch $(target_path)/__init__.py


##@ Container Build Tasks

.PHONY: build
build: ##
	DOCKER_BUILDKIT=1 docker build \
	--file Dockerfile.ubi \
	--target $(server_image_target) \
	--progress plain \
	--tag "$(server_image_name)" .
	docker images
