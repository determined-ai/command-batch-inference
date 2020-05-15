# command-batch-inference

## Prerequisites
Train the model in the `model` directory
```commandline
det e create model/train.yaml model
```
 

## Running batch inference locally
```commandline
pip install -r requirements.txt

python batch.py --master-url <master-url> --experiment-id <experiment-id> --input-path <s3-path-to-input-file> --cpu
```

## Running batch inference on determined
```commandline
det command run --config environment.image=<inference-image> python /batch.py --experiment-id <experiment-id> --input-path <s3-path-to-input-file>
```


Example: 
```commandline
det command run --config environment.image=seanr3215/batch-inference python /batch.py --experiment-id 1 --input-path s3://sean-test-determined-bucket/input/input.jpg
```
