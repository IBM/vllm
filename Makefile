
target_path := "vllm/entrypoints/grpc/pb"

gen-protos:
	# Compile protos
	pip install grpcio-tools==1.60.1 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir $(target_path) || true
	python -m grpc_tools.protoc -Iproto --python_out=$(target_path) \
		--grpc_python_out=$(target_path) --mypy_out=$(target_path) proto/generation.proto
	find $(target_path)/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch $(target_path)/__init__.py

