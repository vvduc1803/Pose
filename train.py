import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset.linemod.linemod import LinemodDataset
from lib.model.mtgope import MTGOPE
from lib.losses import losses_compute
from lib.evaluators.linemod.metrics import Evaluator
from tqdm import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--resume', type=bool, default=False, help='resume training from checkpoint')
    return parser.parse_args()


def train(model, device, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    seg_loss_list = []
    vote_loss_list = []
    depth_loss_list = []

    with tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch") as bar:
        for batch_idx, data in enumerate(bar):
            img, gt_seg, gt_vertex, gt_sparse_depth = data['inp'], data['mask'], data['vertex'], data['depth']
            img = img.to(device)
            gt_seg = gt_seg.to(device)
            gt_vertex = gt_vertex.to(device)
            gt_sparse_depth = gt_sparse_depth.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            pred_seg = outputs['seg']
            pred_vertex = outputs['vertex']
            pred_dense_depth = outputs['depth']

            seg_loss, vote_loss, depth_loss, loss = losses_compute(pred_seg, gt_seg, pred_vertex, gt_vertex, pred_dense_depth,
                                                gt_sparse_depth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            seg_loss_list.append(seg_loss.item())
            vote_loss_list.append(vote_loss.item())
            depth_loss_list.append(depth_loss.item())

            bar.set_postfix({
                'Seg Loss': f"{seg_loss:.4f}",
                'Vote Loss': f"{vote_loss:.4f}",
                'Depth Loss': f"{depth_loss:.4f}",
                'Total Loss': f"{loss.item():.4f}"
            })

    avg_loss = total_loss / len(loader)
    avg_seg_loss = sum(seg_loss_list) / len(seg_loss_list)
    avg_vote_loss = sum(vote_loss_list) / len(vote_loss_list)
    avg_depth_loss = sum(depth_loss_list) / len(depth_loss_list)

    return avg_loss, avg_seg_loss, avg_vote_loss, avg_depth_loss


def test(model, device, loader, epoch):
    model.eval()
    evaluator = Evaluator()

    with tqdm(loader, desc=f"Test Epoch {epoch + 1}", unit="batch") as bar:
        for batch_idx, data in enumerate(bar):
            img, gt_seg, gt_vertex, gt_sparse_depth = data['inp'], data['mask'], data['vertex'], data['depth']
            img = img.to(device)
            gt_seg = gt_seg.to(device)
            gt_vertex = gt_vertex.to(device)
            gt_sparse_depth = gt_sparse_depth.to(device)

            with torch.no_grad():
                outputs = model(img)
                pred_seg = outputs['seg']
                pred_dense_depth = outputs['depth']

                evaluator.evaluate(outputs, data)

            bar.set_postfix(loss="N/A")

    metrics = evaluator.summarize()
    return metrics


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return None


def main():
    args = parse_args()
    device = 'cpu'

    model = MTGOPE()
    model.to(device)

    train_dataset = LinemodDataset('/home/ana/Study/Pose/clean-pvnet/data/linemode/cat/train.json')
    test_dataset = LinemodDataset('/home/ana/Study/Pose/clean-pvnet/data/linemode/cat/train.json')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_path, 'last.pth')
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if start_epoch is None:
            start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        avg_loss, avg_seg_loss, avg_vote_loss, avg_depth_loss = train(model, device, train_data_loader, optimizer,
                                                                      epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_path, f'epoch_{epoch + 1}.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

        # Save last checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(args.checkpoint_path, 'last.pth'))

        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}, Vote Loss: {avg_vote_loss:.4f}, Depth Loss: {avg_depth_loss:.4f}")

        # Test
        metrics = test(model, device, test_data_loader, epoch)
        print("Test Metrics:", metrics)

        # Save metrics and losses
        with open(os.path.join(args.checkpoint_path, 'train_losses.txt'), 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}, Vote Loss: {avg_vote_loss:.4f}, Depth Loss: {avg_depth_loss:.4f}\n")
        with open(os.path.join(args.checkpoint_path, 'test_metrics.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}, Metrics: {metrics}\n")


if __name__ == '__main__':
    main()
