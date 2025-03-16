import torch
import torch.nn as nn
from lib.model.mtgope import MTGOPE
from lib.dataset.linemod.linemod import LinemodDataset
from lib.visual.visualize import Visualizer
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from lib.utils import img_utils

# Load model
def load_model(checkpoint_path):
    model = MTGOPE()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Load dataset
def load_dataset(ann_file):
    dataset = LinemodDataset(ann_file)
    return dataset


# Visualize results
def visualize_results(model, dataset):
    visualizer = Visualizer()
    for index in range(len(dataset)):
        data = dataset.__getitem__(index)
        img = data['inp']
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img)

        visualizer.visualize(outputs, data)


# Main function
def main():
    checkpoint_path = '/home/ana/Study/Pose/checkpoints/epoch_1.pth'
    ann_file = '/home/ana/Study/Pose/clean-pvnet/data/linemode/cat/train.json'

    model = load_model(checkpoint_path)
    model.is_training = False
    dataset = load_dataset(ann_file)

    visualize_results(model, dataset)


if __name__ == '__main__':
    main()
