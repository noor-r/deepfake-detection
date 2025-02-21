import torchvision.models as models

model = models.densenet121(drop_rate=0.2)

print(model)