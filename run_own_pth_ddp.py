
# Nuevo run_own_pth.py adaptado a estructura DDP
import os, argparse, time, datetime, sys, shutil, stat, traceback
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from torchvision.models import ResNet152_Weights
from util.SAR_dataset import SAR_dataset
from util.util_SAR import compute_results, visualize
#from model.CrissCrossAttention_dual_2_sinINF import FEANet
from model.FEANet import FEANet
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Distributed Test with PyTorch')
parser.add_argument('--model_name', '-m', type=str, default='FEANet')
parser.add_argument('--weight_name', '-w', type=str, default='FEANet')
parser.add_argument('--file_name', '-f', type=str, default='190.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--n_class', '-nc', type=int, default=12)
parser.add_argument('--data_dir', '-dr', type=str, default='/workspace/RGB_T_SAR/proyecto_oct_2025_sam2/dataset/dataset_2025/articulo_mdpi')
parser.add_argument('--model_dir', '-wd', type=str, default='/workspace/RGB_T_SAR/CPGFANet/runs_SAR_scenes_vers_feanet_2/FEANet/')
args = parser.parse_args()

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

def main_worker(rank, world_size):
    setup(rank, world_size)
    print(f"[Rank {rank}] Initialized")

    model_path = os.path.join(args.model_dir, args.file_name)
    if not os.path.exists(model_path):
        print(f"[Rank {rank}] ❌ Model file not found: {model_path}")
        cleanup()
        return

    weights = ResNet152_Weights.IMAGENET1K_V1
    resnet1 = models.resnet152(weights)
    resnet2 = models.resnet152(weights)
    model = FEANet(args.n_class, resnet1, resnet2).to(rank)

    checkpoint = torch.load(model_path, map_location=f"cuda:{rank}")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(strip_module_prefix(state_dict))

    model = DDP(model, device_ids=[rank], output_device=rank)

    test_dataset = SAR_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class), dtype=np.int64)
    ave_time_cost = 0.0

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = images.cuda(rank, non_blocking=True)
            labels = labels.cuda(rank, non_blocking=True)
            start_time = time.time()
            logits = model(images)
            end_time = time.time()

            if it >= 5:
                ave_time_cost += (end_time - start_time)

            pred = logits.argmax(1).cpu().numpy().squeeze().flatten()
            true = labels.cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(true, pred, labels=list(range(args.n_class)))
            conf_total += conf

            if rank == 0:
                os.makedirs(f'./result/Pred/{args.weight_name}/', exist_ok=True)
                visualize(image_name=names, predictions=logits.argmax(1), weight_name='Pred1_' + args.weight_name)
                print(f"[{args.model_name}] Frame {it+1}/{len(test_loader)}, Time: {(end_time - start_time)*1000:.2f} ms")

    if dist.is_initialized():
        conf_tensor = torch.tensor(conf_total, dtype=torch.float32, device=f'cuda:{rank}')
        dist.all_reduce(conf_tensor, op=dist.ReduceOp.SUM)
        conf_total = conf_tensor.cpu().numpy()

    if rank == 0:
        recall, iou, _ = compute_results(conf_total)
        print("\n=== Evaluation Results ===")
        for i, (r, iou_val) in enumerate(zip(recall, iou)):
            print(f"Class {i}: Recall = {r:.4f}, IoU = {iou_val:.4f}")
        print(f"Mean Recall: {np.nanmean(recall):.4f}, Mean IoU: {np.nanmean(iou):.4f}")
        print(f"Average Inference Time per Frame: {ave_time_cost * 1000 / (len(test_loader) - 5):.2f} ms")

    cleanup()

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    main_worker(rank, world_size)