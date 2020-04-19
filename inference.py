import glob
from PIL import Image
import torch

import torch.nn as nn
from torchvision import datasets, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes=('with_mask','without_mask')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(

            nn.Conv2d(3, 6, 5),  # (N, 3, 224, 224) -> (N,  6, 220, 220)
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 220, 220) -> (N,  6, 110, 110)
            nn.Conv2d(6, 16, 5),  # (N, 6, 110, 110) -> (N, 16, 106, 106)
            #nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)  # (N,16, 106, 106) -> (N, 16, 53, 53)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(44944, 120),  # (N, 400) -> (N, 120)
            #nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(120, 84),  # (N, 120) -> (N, 84)
            nn.Tanh(),
            nn.Linear(84, 2)  # (N, 84)  -> (N, 2)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # -1, in_features x.size(0), -1
        x = self.fc_model(x)
        return x
model = LeNet()
#net.load_state_dict(torch.load(PATH))
model=torch.load("models/mask_model_lenet_exp2.pth")
model.eval()



def evaluation(dataloader):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).double().sum().item()
    return 100 * correct / total




test_dir = 'data/Test/'
test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

testset = datasets.ImageFolder(test_dir,transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

print(evaluation(testloader))