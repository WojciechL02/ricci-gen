import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import train
from src.architecture import GeometricFlowNet, MetricDecoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparams
    batch_size = 64
    lr = 2e-4
    epochs = 1

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    flow_net = GeometricFlowNet().to(device)
    decoder = MetricDecoder().to(device)

    opt_flow = torch.optim.Adam(flow_net.parameters(), lr=lr)
    opt_decoder = torch.optim.Adam(decoder.parameters(), lr=lr)

    # Initialize TensorBoard Writer
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/grfm_v2_experiment")
    print("TensorBoard Writer initialized. Run 'tensorboard --logdir runs' to view.")

    train(flow_net, decoder, opt_flow, opt_decoder, batch_size, device, epochs, loader, writer)

    writer.close()


if __name__ == "__main__":
    main()
