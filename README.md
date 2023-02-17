# Remote LLM

This is a simple wrapper for a large language model that allows us to call it remotely using gRPC through betterproto. There's a simple server that provides GPTNeo-1.3b too. If you just need a basic client install just the client requirements, `client_requirements.txt`. 

## Development

To regenerate the protobuf interface from the `llm_rpc.proto` run this:
```bash
python -m grpc_tools.protoc -I . --python_betterproto_out=remote_llm llm_rpc.proto
```

## Testing

First start the service LLM with `python tests/test_llm.py`, then run the tests with `pytest`