FROM determinedai/environments:cuda-10.0-pytorch-1.4-tf-1.14-gpu-4bd937a

COPY batch.py /batch.py

COPY determined-wheels/determined_common-0.12.4.dev0-py3-none-any.whl /determined_common-0.12.4.dev0-py3-none-any.whl
COPY determined-wheels/determined-0.12.4.dev0-py3-none-any.whl /determined-0.12.4.dev0-py3-none-any.whl

RUN pip install /determined_common-0.12.4.dev0-py3-none-any.whl
RUN pip install /determined-0.12.4.dev0-py3-none-any.whl
