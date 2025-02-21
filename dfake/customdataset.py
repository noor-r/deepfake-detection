import torch
import torch.utils
import torchvision.datasets as datasets  # Import for datasets
from torchvision.transforms import ToTensor, Compose  # Import for transformations
import torchvision.models as models
from pytorch_model_summary import summary
from model.attention.triplet_attention import TripletAttention  # Import TripletAttention
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from skimage import io, transform
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt


# Define transformations (adjust based on your image format and needs)
transform = transforms.Compose([transforms.Resize((128,128)),
                     transforms.ToTensor(),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomVerticalFlip()])  # Convert to tensors


# train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Define data loader for training (adjust batch size and other parameters as needed)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)


# aa = torch.rand(4,3,128,128)

# print(summary(densenet, aa))

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((128,128)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip()
     ])

data_dir = "D:/deepfake/"
csv_file = 'metadata4.csv'
root_dir = 'faces/'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # self.image_list = os.listdir(root_dir)

        # self.allImageNames = 
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # print("fetching image : ", self.image_list[idx])

        imageName = self.landmarks_frame['videoname'][idx]
        # print('Name of current image is : ', imageName)
        # print('--------------')
        image = Image.open(self.root_dir + imageName)
        
        videoname_to_find = imageName
        


        label = self.landmarks_frame[self.landmarks_frame["videoname"] == videoname_to_find]["label"].values[0]

        # print(f"Label for videoname '{videoname_to_find}': {label}")
        # print('------------')


        if self.transform:
            image = self.transform(image)
        
        if(label == 'FAKE'):
            label = 1
        else:
            label = 0
        

        return (image, label)
    



# for i, data in enumerate(train_loader):


#     images, labels = data

#     print('shape of images : ', images.shape)
#     print('shape of labels : ', labels.shape)

#     if(i == 5):
#         break


# dataset_length = len(face_dataset)
# print(f"Dataset length: {dataset_length}")
# print('Number of batches : ', len(train_loader))

# # Choose an index for __getitem__ function
# sample_idx = 0

# # Call __getitem__ function and print the output
# sample_item = face_dataset[sample_idx]

# print(f"Sample item keys: {list(sample_item.keys())}")  # Print keys of the sample item
# print(f"Sample item image shape: {sample_item['image'].shape}")  # Assuming 'image' key is used in __getitem__
# face_dataset = FaceLandmarksDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
# batch_size = 4
# data_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)




class Model(nn.Module):

    def __init__(self, densenet):

        super().__init__()

        self.features = densenet.features
        self.classifier = densenet.classifier

        for param in self.parameters():
            param.requires_grad = False

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

        # print('shape before mp1 : ', x.shape)
        x1 = self.mp1(x)
        # print('shape after mp1 : ', x1.shape)

        x = self.features.denseblock1(x)
        x = self.features.transition1(x)

        # print('shape before mp2 : ', x.shape)
        x2 = self.mp2(x)
        # print('shape after mp2 : ', x2.shape)


        x = self.features.denseblock2(x)
        x = self.features.transition2(x)

        # print('shape before mp2 : ', x.shape)
        x3 = self.mp3(x)
        # print('shape after mp2 : ', x2.shape)

        x = self.features.denseblock3(x)
        x = self.features.transition3(x)

        x4 = x

        x = self.features.denseblock4(x)

        x = self.features.norm5(x)

        x5 = x

        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)

        x_new = torch.cat((x1,x2,x3,x4,x5), 1)

        x_final = self.conv1x1(x_new)

        x_final = self.attention(x_final)

        # print('shape of concatenated : ', x_final.shape)

        features = x_final
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)


        return out
    
densenet = models.densenet121(weights = "DenseNet121_Weights.IMAGENET1K_V1")
model = Model(densenet)
model.classifier

print("summary of the proposed model")
print(summary(model, torch.rand(4,3,128,128)))

model.to(device)

face_dataset = FaceLandmarksDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
print('total length of dataset : ', len(face_dataset))
trainset, validationset, testset = torch.utils.data.random_split(face_dataset, [16000,1153,1000])
# trainset,testset=torch.utils.data.random_split(face_dataset, [40,20])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size = 64, shuffle=True)

print(len(face_dataset))

num_batches = len(train_loader)
val_num_batches = len(validation_loader)

print('number of batches in training set : ', len(train_loader))



import torch.optim as optim
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

training_losses = []
validation_losses = []
validation_accuracy = []

total_epochs = 75

for epoch in range(total_epochs):  # loop over the dataset multiple times

    epoch_start = datetime.now()
    running_loss = 0.0
    running_loss2 = 0.0

    val_running_loss = 0.0
    val_running_loss2 = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # print(i)

        images = images.to(device)
        labels = labels.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_loss2 += loss.item()

        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.6f}')
            running_loss = 0.0
        
    print('--------------')
    print('Validating')
    print('-------------')
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for j, data in enumerate(validation_loader):
            
            
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            val_running_loss2 += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()



            if j % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {j + 1:5d}] loss: {val_running_loss / 10:.6f}')
                val_running_loss = 0.0




    
    scheduler.step()
    # print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    loss_per_epoch = running_loss2 / num_batches
    training_losses.append(loss_per_epoch)

    val_loss_per_epoch = val_running_loss2 / val_num_batches
    validation_losses.append(val_loss_per_epoch)

    
    print(correct_val)
    print(total_val)
   
    accuracy_val = 100 * correct_val / total_val
    validation_accuracy.append(accuracy_val)

    epoch_end = datetime.now()
    print("Epoch time : ", (epoch_end - epoch_start).total_seconds())
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy_val:.2f} %')
print('Finished Training')

print('loss values for each epoch')
print(training_losses)
print('accuracy values for each epoch')
print(validation_accuracy)

import matplotlib.pyplot as plt
import numpy as np
 


# a = torch.rand(4, 128, 4, 4)
# b = torch.rand(4, 256, 4, 4)

# c = torch.cat((a,b), 1)
# c.shape
model.eval()
correct = 0
total = 0

precision_scores_test = []
recall_scores_test = []
auc_scores_test = []
fpr_list_test = []
tpr_list_test = []

model.eval()
correct_test = 0
total_test = 0
all_predictions_test = []
all_labels_test = []
all_probs_test = []  # To store probabilities

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        # total_test += labels.size(0)
        # correct_test += (predicted == labels).sum().item()

        # Collect predictions and labels for calculating metrics
        all_predictions_test.extend(predicted.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())
        all_probs_test.extend(probs.cpu().numpy())

# Calculate accuracy for the test set
accuracy_test = accuracy_score(all_labels_test, all_predictions_test)

# Calculate precision, recall, AUC
precision_test = precision_score(all_labels_test, all_predictions_test)
recall_test = recall_score(all_labels_test, all_predictions_test)

# Calculate AUC using probabilities
auc_test = roc_auc_score(all_labels_test, np.array(all_probs_test)[:, 1])  # Assuming binary classification

# Calculate ROC curve for test set
fpr_test, tpr_test, _ = roc_curve(all_labels_test, np.array(all_probs_test)[:, 1])  # Assuming binary classification




# Store metrics in respective arrays
precision_scores_test.append(precision_test)
recall_scores_test.append(recall_test)
auc_scores_test.append(auc_test)
fpr_list_test.append(fpr_test)
tpr_list_test.append(tpr_test)

print(f'Test Accuracy: {accuracy_test:.2f} %')
print(f'Test Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, AUC: {auc_test:.4f}')

# Calculate F1 score
f1_test = f1_score(all_labels_test, all_predictions_test)

# Print F1 score
print(f'Test F1 Score: {f1_test:.4f}')






# define data values
x = range(total_epochs) # X-axis points
y1 = np.array(training_losses)  # Y-axis points
y2 = np.array(validation_losses)

plt.plot(x, y1) 
plt.plot(x, y2)  

plt.show()

# Plotting accuracy vs epoch
plt.figure(figsize=(10, 5))
plt.plot(range(total_epochs), validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.4f)' % auc_test)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC score
print(f'AUC Score: {auc_test:.4f}')

# bar chart
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy_test, precision_test, recall_test, f1_test]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'green', 'orange', 'purple'])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.ylim([0, 1])  # Set y-axis limit to range [0, 1]
plt.show()