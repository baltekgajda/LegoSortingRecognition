import os
import datetime
import torch


def save_model(model, models_dir):
    user = os.getlogin()
    time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filepath = models_dir + "/model-" + user + "-" + time + ".pth"
    torch.save(model, filepath)

