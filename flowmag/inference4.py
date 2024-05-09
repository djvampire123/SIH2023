import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import concurrent.futures

from dataset import TrainingFramesDataset, FramesDataset
from test_time_adapt import test_time_adapt
from myutils import get_our_model, write_video, dist_transform

def load_and_process_mask(mask_path, soft_mask):
    '''
    Load and preprocess a single mask
    '''
    mask = np.load(mask_path)
    mask = torch.tensor(mask)
    mask = torch.squeeze(mask)
    h, w = mask.shape
    mask = mask.float()

    if soft_mask:
        print('Softening mask')
        dist = dist_transform(mask)
        dist[dist < soft_mask] = 1
        dist[dist >= soft_mask] = 0
        mask = dist

    return mask

def inference(model, frames_dataset, save_dir, alpha, max_alpha, mask, num_device, output_video):
    device = 'cuda'
    save_dir.mkdir(exist_ok=True, parents=True)
    results = []

    if isinstance(model, nn.Module):
        model.eval()
    
    training_status = model.module.get_training_status()

    # If input alpha exceeds the range for training, perform recursions for inference
    if alpha > max_alpha and np.sqrt(alpha) < max_alpha:
        our_alpha = np.sqrt(alpha)
        num_recursion = 2
    elif alpha < max_alpha:
        our_alpha = alpha
        num_recursion = 1
    else:
        raise Exception('alpha out of range')

    with torch.no_grad():
        im0 = frames_dataset[0][None].to(device)
        results.append(im0.detach().cpu())

        for i in tqdm(range(1, len(frames_dataset))):
            # Get i^th frame, and merge with 0^th frame
            im1 = frames_dataset[i][None].to(device)
            frames = torch.stack([im0, im1], dim=2).repeat(num_device,1,1,1,1)

            # Process frames
            for _ in range(num_recursion):
                if training_status:
                    pred, _, _ = model(frames, alpha=our_alpha, mask=mask)
                else:
                    pred = model(frames, alpha=our_alpha, mask=mask)
                frames = torch.stack([im0, pred[0,:,0].unsqueeze(0)], dim=2).repeat(num_device,1,1,1,1)

            # Save predicted frame
            pred = pred[0,:,0]
            results.append(pred.detach().cpu())

    # Save as video file
    if output_video:
        saved_frames = [(255*img.squeeze().permute(1,2,0).flip([-1]).numpy()).astype(np.uint8) for img in results]
        video_path = str(save_dir / f'x{alpha}.mp4') if mask is None else str(save_dir / f'masked_x{alpha}.mp4')
        write_video(saved_frames, 30, video_path)
        print('saved the video to {}'.format(video_path))
    
    # Save as image files
    else:
        save_dir = save_dir / f'x{alpha}' if mask is None else save_dir / f'masked_x{alpha}'
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(results):
            save_image(img, save_dir / f'{i+1:04}.png')
        print('saved the images to {}'.format(str(save_dir)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--frames_dir', type=str, required=True, help='path to directory of frames to magnify')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--save_name', type=str, required=True, help='name to save under')
    parser.add_argument('--alpha', type=float, required=True, help='amount to magnify motion')
    parser.add_argument('--mask_paths', nargs='+', help='paths to numpy masks, if empty then no mask')
    parser.add_argument('--soft_mask', type=int, default=0, help='how much to soften mask. 0 is none, higher is more')
    parser.add_argument('--output_video', action='store_true')
    parser.add_argument('--test_time_adapt', action='store_true')
    parser.add_argument('--tta_epoch', type=int, default=3, help='number of epochs for test time adaptation')

    args = parser.parse_args()

    # ... [rest of your code for loading frames, models, etc.] ...

    frames_dataset = TrainingFramesDataset(args.frames_dir) if args.test_time_adapt else FramesDataset(args.frames_dir)
    config = OmegaConf.load(args.config)
    max_alpha = config.train.alpha_high
    save_dir = Path(args.resume).parent.parent / 'inference' / args.save_name
    save_dir.mkdir(exist_ok=True, parents=True)
    model, epoch = get_our_model(args, args.test_time_adapt)

    # Load and preprocess all masks
    masks = []
    if args.mask_paths:
        for mask_path in args.mask_paths:
            mask = load_and_process_mask(mask_path, args.soft_mask)
            masks.append(mask)

    # Parallel processing for each mask
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for mask in masks:
            future = executor.submit(inference, model, frames_dataset, save_dir, args.alpha, max_alpha, mask, 1, args.output_video)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    # If no masks are provided, run inference without mask
    if not masks:
        inference(model, frames_dataset, save_dir, args.alpha, max_alpha, None, 1, args.output_video)
