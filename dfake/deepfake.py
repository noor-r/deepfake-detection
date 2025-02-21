import torch
import torchvision.datasets as datasets  # Import for datasets
from torchvision.transforms import ToTensor, Compose  # Import for transformations
import torchvision.models as models
from pytorch_model_summary import summary
from model.attention.triplet_attention import TripletAttention  # Import TripletAttention
import torchvision.transforms as transforms

# from model.attention.TripletAttention import TripletAttention (assuming it's defined elsewhere)
data_dir = "D:\deepfake"

# Define transformations (adjust based on your image format and needs)
transform = Compose([ToTensor()])  # Convert to tensors
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Define data loader for training (adjust batch size and other parameters as needed)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

densenet = models.densenet121()
aa = torch.rand(4,3,128,128)

print(summary(densenet, aa))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, densenet):

        super().__init__()

        self.features = densenet.features
        self.classifier = densenet.classifier

        self.mp1 = nn.MaxPool2d(8,8)
        self.mp2 = nn.MaxPool2d(4,4)
        self.mp3 = nn.MaxPool2d(2,2)

        self.conv1x1 = nn.Conv2d(1984, 1024, 1)

        self.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

        self.attention = TripletAttention()

    def forward(self, x):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        print('shape before mp1 : ', x.shape)
        x1 = self.mp1(x)
        print('shape after mp1 : ', x1.shape)

        x = self.features.denseblock1(x)
        x = self.features.transition1(x)

        print('shape before mp2 : ', x.shape)
        x2 = self.mp2(x)
        print('shape after mp2 : ', x2.shape)


        x = self.features.denseblock2(x)
        x = self.features.transition2(x)

        print('shape before mp2 : ', x.shape)
        x3 = self.mp3(x)
        print('shape after mp2 : ', x2.shape)

        x = self.features.denseblock3(x)
        x = self.features.transition3(x)

        x4 = x

        x = self.features.denseblock4(x)

        x = self.features.norm5(x)

        x5 = x

        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)
        print(x5.shape)

        x_new = torch.cat((x1,x2,x3,x4,x5), 1)

        x_final = self.conv1x1(x_new)

        x_final = self.attention(x_final)

        print('shape of concatenated : ', x_final.shape)

        features = x_final
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)


        return out
model = Model(densenet)
model.classifier
print(summary(model, aa))

a = torch.rand(4, 128, 4, 4)
b = torch.rand(4, 256, 4, 4)

c = torch.cat((a,b), 1)
c.shape
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)