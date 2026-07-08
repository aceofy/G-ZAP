import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import scipy.io as sio 
import os
import time


from models.model_INR import BF_NIR_conv_Cell 
# from models.model_SDE import Net_ms2pan
import wald_utilities
from loss import LossCalculator
import utils

utils.set_seed()


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="H5文件路径")
parser.add_argument("--id", type=int, default=0, help="起始图片 ID (Start ID)")
parser.add_argument("--num_images", type=int, default=-1, help="要测试的图片数量 (-1 代表一直运行到文件结束)")
parser.add_argument("--epochs", type=int, default=500, help="每张图的训练轮数")
parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
parser.add_argument("--save_dir", type=str, default="results_mat", help="结果保存路径")
parser.add_argument("--sensor", type=str, default="WV3", help="sensor type used by Wald degradation and saved metadata")
parser.add_argument("-n", "--eval_scale", type=float, default=1, help="quantitative evaluation scale relative to original PAN")


parser.add_argument("--w_level2", type=float, default=1.)
parser.add_argument("--w_level1", type=float, default=1.)
parser.add_argument("--w_level0", type=float, default=4)

args = parser.parse_args()
if args.eval_scale <= 0:
    raise ValueError("--eval_scale/-n must be greater than 0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available() and not torch.cuda.is_available():
    device = torch.device("mps")
print(f"Using Device: {device}")


def get_cell(H, W, device):
    """ 生成 LIIF 风格的 Cell Tensor (B, H*W, 2) """
    hy = 2 / H
    hx = 2 / W
    cell = torch.tensor([hx, hy], dtype=torch.float32, device=device)
    return cell.unsqueeze(0).unsqueeze(0).repeat(1, H*W, 1)

def upsample(x, size):
    """ 双三次插值上采样 """
    return F.interpolate(x, size=size, mode='bicubic', align_corners=False)

def resize_for_eval(x, eval_scale):
    if eval_scale == 1:
        return x
    return F.interpolate(x, scale_factor=1 / eval_scale, mode='bicubic', align_corners=False)

def format_scale(eval_scale):
    if float(eval_scale).is_integer():
        return str(int(eval_scale))
    return f"{eval_scale:g}"

def get_crops(pan, lms, ms, gt):
    """ 生成 6 对训练数据：1 个全图 + 5 个 Crops """
    crops_pan, crops_lms, crops_ms, crops_gt = [], [], [], []
    
    
    crops_pan.append(pan)
    crops_lms.append(lms)
    crops_ms.append(ms)
    crops_gt.append(gt)
    
    B, _, H, W = pan.shape
    h_c, w_c = H // 2, W // 2
    
    
    _, _, H_ms, W_ms = ms.shape
    h_ms_c, w_ms_c = H_ms // 2, W_ms // 2

    
    positions = [
        (0, 0), (0, w_c), (h_c, 0), (h_c, w_c), (h_c // 2, w_c // 2)
    ]
    positions_ms = [
        (0, 0), (0, w_ms_c), (h_ms_c, 0), (h_ms_c, w_ms_c), (h_ms_c // 2, w_ms_c // 2)
    ]

    # for (y, x), (y_m, x_m) in zip(positions, positions_ms):
    #     crops_pan.append(pan[..., y:y+h_c, x:x+w_c])
    #     crops_lms.append(lms[..., y:y+h_c, x:x+w_c])
    #     crops_gt.append(gt[..., y:y+h_c, x:x+w_c])
    #     crops_ms.append(ms[..., y_m:y_m+h_ms_c, x_m:x_m+w_ms_c])

    return crops_pan, crops_lms, crops_ms, crops_gt

def save_inference_results(output_tensor, ref_tensor, ms_tensor, pan_tensor, save_dir, data_id,
                           eval_scale, sensor, gt_tensor=None, lms_tensor=None):
    os.makedirs(save_dir, exist_ok=True)
    def to_numpy_HWC(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    save_dict = {
        'I_MS': to_numpy_HWC(ref_tensor),
        'proposed': to_numpy_HWC(output_tensor),
        'I_MS_LR': to_numpy_HWC(ms_tensor),           
        'I_PAN': to_numpy_HWC(pan_tensor),         
        'ratio': 4,
        'eval_scale': eval_scale,
        'sensor_type': sensor.upper()
    }
    if gt_tensor is not None:
        save_dict['gt'] = to_numpy_HWC(gt_tensor)
        if lms_tensor is not None:
            save_dict['I_MS_UP'] = to_numpy_HWC(lms_tensor)

    if eval_scale == 1:
        file_name = f"{data_id}_best.mat"
    else:
        file_name = f"{data_id}X{format_scale(eval_scale)}_best.mat"
    file_path = os.path.join(save_dir, file_name)
    sio.savemat(file_path, save_dict)
    print(f"Saved: {file_path}")


def run_batch_process():
    
    with h5py.File(args.path, 'r') as f:
        total_samples = f['lms'].shape[0]
        print(f"Dataset total samples: {total_samples}")

    
    start_id = args.id
    if args.num_images == -1:
        end_id = total_samples
    else:
        end_id = min(start_id + args.num_images, total_samples)

    print(f"Processing range: ID {start_id} to {end_id - 1}")
    print(f"Sensor: {args.sensor}")
    print(f"Evaluation scale: {args.eval_scale} (train inputs are resized by 1/{args.eval_scale}; saved tensors keep original size)")

    
    for curr_id in range(start_id, end_id):
        print(f"\n{'='*20} Processing Image ID: {curr_id} {'='*20}")
        
        
        with h5py.File(args.path, 'r') as f:
            lms_0 = torch.from_numpy(f['lms'][curr_id]).unsqueeze(0).float().to(device)
            ms_0  = torch.from_numpy(f['ms'][curr_id]).unsqueeze(0).float().to(device)
            pan_0 = torch.from_numpy(f['pan'][curr_id]).float().to(device)
            gt_0 = None
            for gt_key in ("gt", "GT"):
                if gt_key in f:
                    gt_0 = torch.from_numpy(f[gt_key][curr_id]).unsqueeze(0).float().to(device)
                    break
            
            if pan_0.dim() == 2: pan_0 = pan_0.unsqueeze(0).unsqueeze(0)
            elif pan_0.dim() == 3: pan_0 = pan_0.unsqueeze(0)
            lms_full = lms_0
            ms_full = ms_0
            pan_full = pan_0
            gt_full = gt_0

            lms_0 = resize_for_eval(lms_full, args.eval_scale)
            ms_0 = resize_for_eval(ms_full, args.eval_scale)
            pan_0 = resize_for_eval(pan_full, args.eval_scale)
            pan_0_3c = pan_0.repeat(1, 8, 1, 1) 

        
        with torch.no_grad():
            # Level 1 (Pan 128, MS 32)
            ms_1 = wald_utilities.wald_protocol_v1(ms_0, pan_0, 4, args.sensor, 8)
            pan_1 = wald_utilities.wald_protocol_v2(ms_0, pan_0, 4, args.sensor, 8)
            lms_1 = upsample(ms_1, size=pan_1.shape[-2:]) 
            pan_1_3c = pan_1.repeat(1, 8, 1, 1)

            # Level 2 (Pan 32, MS 8)
            ms_2 = wald_utilities.wald_protocol_v1(ms_1, pan_1, 4, args.sensor, 8)
            pan_2 = wald_utilities.wald_protocol_v2(ms_1, pan_1, 4, args.sensor, 8)
            lms_2 = upsample(ms_2, size=pan_2.shape[-2:])
            pan_2_3c = pan_2.repeat(1, 8, 1, 1)

        data_dict = {
            'L0': {'ms': ms_0, 'lms': lms_0, 'pan': pan_0_3c},
            'L1': {'ms': ms_1, 'lms': lms_1, 'pan': pan_1_3c, 'gt': ms_0},
            'L2': {'ms': ms_2, 'lms': lms_2, 'pan': pan_2_3c, 'gt': ms_1, 'gt_enhanced': ms_0}
        }

        
        
        H0, W0 = lms_0.shape[-2:]
        cell_L0 = get_cell(H0, W0, device)

        
        model = BF_NIR_conv_Cell(feat_dim=64, guide_dim=64,spa_edsr_num=6, mlp_dim=[256,128],NIR_dim=48, cell_decode=True).to(device)

        # model_sde = Net_ms2pan().to(device)
        # model_sde.load_state_dict(torch.load(r"D:\vscode_file_folder\ZS-Pan\model_SDE\wv3\0_Net_ms2pan.pth", map_location=device, weights_only=True))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
        loss_calculator = LossCalculator(sensor=args.sensor, ratio=4, device=device)
        l1_loss = nn.L1Loss()

        
        model.train()
        best_loss = float('inf')
        
        weight_dir = "temp_weights"
        os.makedirs(weight_dir, exist_ok=True)
        temp_pth_name = os.path.join(weight_dir,f"temp_best_model_id{curr_id}.pth")

        start_time = time.time()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            # 1. Level 2 Tasks
            
            target_shape_L0_MS = data_dict['L0']['ms'].shape[-2:] # (128, 128)
            cell_L2_enhanced = get_cell(target_shape_L0_MS[0], target_shape_L0_MS[1], device)
            out_L2_enh = model(data_dict['L2']['pan'], data_dict['L2']['lms'], None, 
                           cell=cell_L2_enhanced, target_shape=target_shape_L0_MS)
            loss_L2 = l1_loss(out_L2_enh, data_dict['L0']['ms'])
            
            target_shape_L1_MS = data_dict['L1']['ms'].shape[-2:] # (32, 32)
            cell_L2_reg = get_cell(target_shape_L1_MS[0], target_shape_L1_MS[1], device)
            out_L2_reg = model(data_dict['L2']['pan'], data_dict['L2']['lms'], None, 
                           cell=cell_L2_reg, target_shape=target_shape_L1_MS)
            loss_L2 += l1_loss(out_L2_reg, data_dict['L1']['ms'])
            loss_L2 = loss_L2 / 2
            
            # 2. Level 1 Augmented Tasks (Crops)
            c_pans, c_lmss, c_mss, c_gts = get_crops(data_dict['L1']['pan'], data_dict['L1']['lms'], data_dict['L1']['ms'], data_dict['L1']['gt'])
            loss_L1 = 0
            for p, l, m, g in zip(c_pans, c_lmss, c_mss, c_gts):
                target_shape_crop = g.shape[-2:]
                cell_crop = get_cell(target_shape_crop[0], target_shape_crop[1], device)
                out_crop = model(p, l, None, cell=cell_crop, target_shape=target_shape_crop)
                loss_L1 += l1_loss(out_crop, g)
            loss_L1 = loss_L1 / len(c_pans)
            
            # 3. Level 0 Unsupervised (Spectral Loss)
            out_L0 = model(data_dict['L0']['pan'], data_dict['L0']['lms'], None, cell=cell_L0)
            loss_L0 = loss_calculator.compute_spectral_loss(out_L0.squeeze(0).permute(1,2,0), 
                                                            data_dict['L0']['ms'].squeeze(0).permute(1,2,0))
            
            
            # out_L0_from_L1 = model(data_dict['L1']['pan'], data_dict['L1']['lms'], None, cell=cell_L0, target_shape=out_L0.shape[-2:])
            # gamma = 0
            # loss_L0 += gamma * l1_loss(out_L0_from_L1, out_L0)


            # Total Loss
            loss_total = (args.w_level2 * loss_L2) + (args.w_level1 * loss_L1) + (args.w_level0 * loss_L0)
            
            loss_total.backward()
            optimizer.step()

            # Save Best
            current_loss_val = loss_total.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                torch.save(model.state_dict(), temp_pth_name)

            if epoch % 50 == 0: 
                print(f"ID {curr_id} | Epoch {epoch} | Total: {current_loss_val:.5f} (L2:{loss_L2.item():.4f} L1:{loss_L1.item():.4f} L0:{loss_L0.item():.4f})")

        print(f"ID {curr_id} Training Finished. Best Loss: {best_loss:.5f} | Time: {time.time()-start_time:.1f}s")

        
        print("Running Inference...")
        model.load_state_dict(torch.load(temp_pth_name, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            ref_full = gt_full if gt_full is not None else lms_full
            final_shape = ref_full.shape[-2:]
            final_cell = get_cell(final_shape[0], final_shape[1], device)
            final_out = model(
                data_dict['L0']['pan'],
                data_dict['L0']['lms'],
                None,
                cell=final_cell,
                target_shape=final_shape
            )
            save_inference_results(
                final_out,
                ref_full,
                ms_full,
                pan_full,
                args.save_dir,
                curr_id,
                eval_scale=args.eval_scale,
                sensor=args.sensor,
                gt_tensor=gt_full,
                lms_tensor=lms_full
            )
        
        
        # if os.path.exists(temp_pth_name):
        #     os.remove(temp_pth_name)

if __name__ == "__main__":
    run_batch_process()
