import argparse
import deepspeed as ds
import torch
import torch.nn as nn

from model import get_model
from dataset import get_dataset

parser = argparse.ArgumentParser()
parser = ds.add_config_arguments(parser)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher",
)
args = parser.parse_args()

model = get_model()
dataset = get_dataset()

model_engine, optimizer, train_loader, _ = ds.initialize(
    args=args, model=model, model_parameters=model.parameters(), training_data=dataset
)
model_engine.train()

criterion = nn.BCELoss()

num_epochs = 5
running_loss = 0.0
running_correct = 0
running_count = 0
for _ in range(num_epochs):
    for _, data in enumerate(train_loader):
        image, target = data
        n = target.size(0)

        image = image.to(model_engine.local_rank)
        target = target.to(model_engine.local_rank)

        output = model_engine(image)

        loss = criterion(output, target)
        prediction = output >= 0.5

        running_loss += loss.detach() * n
        running_correct += torch.sum(prediction == target)
        running_count += n

        model_engine.backward(loss)
        model_engine.step()