# Remote LLM

This is a simple wrapper for langchain that allows us to call it remotely using gRPC through betterproto. 

To regenerate the protobuf interface from the `llm_rpc.proto` run this:
```bash
python -m grpc_tools.protoc -I . --python_betterproto_out=remote_llm llm_rpc.proto
```

There's a simple server that provides GPTNeo-1.3b too. 

## Testing

First start the service LLM with `python tests/test_llm.py`, then run the tests with `pytest`