import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from dataset import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model = model.to(device)
model.train()

train_dataset = get_dataset()
train_dataloader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=4
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05, weight_decay=1e-07)

num_epochs = 5
running_loss = 0.0
running_correct = 0
running_count = 0
for _ in range(num_epochs):
    for data in tqdm(train_dataloader):
        image, target = data
        n = target.size(0)

        image = image.to(device)
        target = target.to(device)

        output = model(image)

        loss = criterion(output, target)
        prediction = output >= 0.5

        running_loss += loss.detach() * n
        running_correct += torch.sum(prediction == target)
        running_count += n

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
