FROM determinedai/environments:cuda-10.0-pytorch-1.4-tf-1.14-gpu-4bd937a

COPY batch.py /batch.py

RUN pip install determined
