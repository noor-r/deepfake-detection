import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random


df = pd.read_csv('metadata.csv')


filenames = df['videoname'][:10]  
labels = df['label'][:10]        

def fetch_image(filename):
    
    image_path = os.path.join('faces', filename)
    img = Image.open(image_path)
    return img


def preprocess_image(img):
    transform = T.Compose([
       T.Resize((128, 128)),                    
       T.RandomHorizontalFlip(),               
       T.RandomVerticalFlip(),                  
       T.ToTensor(),                            # Convert to tensor
       T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) 
    ])
    return transform(img)


fig, axes = plt.subplots(2, 5, figsize=(20, 8))


for i, (filename, label) in enumerate(zip(filenames, labels)):
    
    img = fetch_image(filename)
    

    img = preprocess_image(img)
    
    
    row = i // 5  
    col = i % 5   
    axes[row, col].imshow(img.permute(1, 2, 0))  # Convert tensor to (H, W, C) format
    axes[row, col].set_title(label)
    axes[row, col].axis('off')                   


plt.tight_layout()
plt.show()
