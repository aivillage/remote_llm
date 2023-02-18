FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /remote_llm
RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt /remote_llm/requirements.txt

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install transformers torch

COPY remote_llm remote_llm/
COPY huggingface_service.py /remote_llm/huggingface_service.py
EXPOSE 50055
# Run the service
CMD ["python3", "huggingface_service.py"]