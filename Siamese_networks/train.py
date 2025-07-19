import torch 
import os
from torch.utils.data import DataLoader
from models.siamese_net import SiameseNet
from utils.data_utils import SiameseMNIST
from config import ConstructiveLoss

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

dataset = SiameseMNIST(train=True)
loader = DataLoader(dataset, batch_size=64,shuffle=True)
model = SiameseNet().to(device)
criterion = ConstructiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(10):
    total_loss = 0
    for img1, img2,label in loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        feat1, feat2 = model(img1,img2)
        loss = criterion(feat1,feat2,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'Saved_model/siamese_model.pth')