import argparse
import pathlib
import time

import boto3
from determined.experimental import Determined

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
from PIL import Image

s3 = boto3.client('s3')


def get_transform():
    transforms = [ToTensor()]
    return Compose(transforms)


def filter_boxes(boxes, scores, threshold=0.9):
    cutoff = 0
    for i, score in enumerate(scores):
        if score < threshold:
            break
        cutoff = i
    return boxes[:cutoff]


def draw_example(image, labels, output, title=None):
    fig, ax = plt.subplots(1)
    plt.title(title)
    ax.imshow(image)
    boxes = labels['boxes'].cpu().numpy()
    boxes = np.vsplit(boxes, boxes.shape[0])
    for box in boxes:
        box = np.squeeze(box)
        bottom, left = box[0], box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle((bottom, left), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # # Add the patch to the Axes
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(output)
    return output


def load_and_transform_image(img_path, cpu):
    image = Image.open(img_path).convert("RGB")
    if cpu:
        image = get_transform()(image)
    else:
        image = get_transform()(image).cuda()
    return image


def main(experiment_id, master=None, input_path=None, cpu=False):
    # Download data
    print('downlading data from', input_path)
    path = input_path.split('s3://')[1]
    bucket = path.split('/')[0]
    key = '/'.join(path.split('/')[1:])
    download_path = pathlib.Path('.') / 'test.jpg'
    s3.download_file(bucket, key, str(download_path))

    # Retrieve model checkpoint
    print('fetching model from experiment', experiment_id)
    d = Determined(master=master)
    checkpoint = d.get_experiment(experiment_id).top_checkpoint()
    if cpu:
        print('loading cpu model')
        model = checkpoint.load(map_location=torch.device('cpu'))
    else:
        print('loading gpu model')
        model = checkpoint.load()

    # Make predictions
    print('making predictions')
    test_image = load_and_transform_image(download_path, cpu)
    with torch.no_grad():
        outputs = model(test_image.unsqueeze(0))[0]

    # Upload results
    output_key = f'output/{time.time()}/output.png'
    print('uploading result to', f's3://{bucket}/{output_key}')
    boxes = filter_boxes(outputs['boxes'], outputs['scores'])
    output = draw_example(test_image.permute(1, 2, 0).cpu().numpy(), {'boxes': boxes}, 'output.png',
                          title="Predictions")
    s3.upload_file(output, bucket, output_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', type=int, required=True)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--master-url', type=str, default=None)
    parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()
    main(args.experiment_id, args.master_url, input_path=args.input_path, cpu=args.cpu)
