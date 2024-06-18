import grpc
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                            ConsoleSpanExporter)
from opentelemetry.trace import SpanKind, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator)

from vllm.entrypoints.grpc.pb import generation_pb2, generation_pb2_grpc

trace_provider = TracerProvider()
set_tracer_provider(trace_provider)

trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace_provider.get_tracer("dummy-client")

with grpc.insecure_channel("localhost:50051") as channel:
    stub = generation_pb2_grpc.GenerationServiceStub(channel)

    with tracer.start_as_current_span("client-span",
                                      kind=SpanKind.CLIENT) as span:
        prompt = "San Francisco is a"
        span.set_attribute("prompt", prompt)

        # Inject the current context into the gRPC metadata
        headers = {}
        TraceContextTextMapPropagator().inject(headers)
        metadata = list(headers.items())

        reqs = [generation_pb2.GenerationRequest(text=prompt, )]

        req = generation_pb2.BatchedGenerationRequest(
            model_id="facebook/opt-125m",
            requests=reqs,
            params=generation_pb2.Parameters(
                sampling=generation_pb2.SamplingParameters(temperature=0.0),
                stopping=generation_pb2.StoppingCriteria(max_new_tokens=10)))
        response = stub.Generate(req, metadata=metadata)
