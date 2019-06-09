import torch


def save_model(model, filename, models_dir='./models'):
    filepath = models_dir + "/" + filename + ".pth"
    torch.save(model, filepath)

