import h5py
import torch
import torch.nn.functional as F
import argparse
import scipy.io as sio
import os

from models.model_INR_band4 import BF_NIR_conv_Cell
import wald_utilities


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="H5 file path")


parser.add_argument("--id", type=int, default=None, help="single sample index (optional)")
parser.add_argument("--start_id", type=int, default=0, help="batch start index (inclusive)")
parser.add_argument("--end_id", type=int, default=None, help="batch end index (inclusive). If None, use --num or last index")
parser.add_argument("--num", type=int, default=20, help="number of samples to run from start_id (optional)")

parser.add_argument("--save_dir", type=str, default="test_band4_results_mat", help="结果保存路径")
parser.add_argument("--sensor", type=str, default="GF2", help="sensor type used by Wald degradation and saved metadata")
parser.add_argument("-n", "--eval_scale", type=float, default=1, help="cell size divisor for inference")
parser.add_argument("--save_pth_name", type=str, default=r"temp_weights_band4\temp_best_model_id0.pth",
                    help="best model path for inference")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_cell(H, W, device):
    hy = 2 / H
    hx = 2 / W
    cell = torch.tensor([hx, hy], dtype=torch.float32, device=device)
    return cell.unsqueeze(0).unsqueeze(0).repeat(1, H * W, 1)

def upsample(x, size):
    return F.interpolate(x, size=size, mode='bicubic', align_corners=False)

def save_inference_results(output_tensor, gt_tensor, ms_tensor, pan_tensor, save_dir, data_id, n_scale, sensor):
    os.makedirs(save_dir, exist_ok=True)

    def to_numpy_HWC(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    save_dict = {
        'I_MS': to_numpy_HWC(gt_tensor),
        'proposed': to_numpy_HWC(output_tensor),
        'I_MS_LR': to_numpy_HWC(ms_tensor),
        'I_PAN': to_numpy_HWC(pan_tensor),
        'ratio': 4,
        'sensor_type': sensor.upper()
    }

    out_path = os.path.join(save_dir, f"{data_id}X{n_scale}_best.mat")
    sio.savemat(out_path, save_dict)
    print(f"[Finished] id={data_id} saved: {out_path}")


def build_data_dict_for_id(f, idx, sensor, n_bands=4):
    """从 h5 文件中读取指定 idx，并生成 L0/L1/L2 wald 数据"""
    lms_0 = torch.from_numpy(f['lms'][idx]).unsqueeze(0).float().to(device)
    ms_0  = torch.from_numpy(f['ms'][idx]).unsqueeze(0).float().to(device)
    pan_0 = torch.from_numpy(f['pan'][idx]).float().to(device)

    if pan_0.dim() == 2:
        pan_0 = pan_0.unsqueeze(0).unsqueeze(0)
    elif pan_0.dim() == 3:
        pan_0 = pan_0.unsqueeze(0)

    pan_0_3c = pan_0.repeat(1, n_bands, 1, 1)

    with torch.no_grad():
        # Level 1
        ms_1 = wald_utilities.wald_protocol_v1(ms_0, pan_0, 4, sensor, n_bands)
        pan_1 = wald_utilities.wald_protocol_v2(ms_0, pan_0, 4, sensor, n_bands)
        lms_1 = upsample(ms_1, size=pan_1.shape[-2:])
        pan_1_3c = pan_1.repeat(1, n_bands, 1, 1)

        # Level 2
        ms_2 = wald_utilities.wald_protocol_v1(ms_1, pan_1, 4, sensor, n_bands)
        pan_2 = wald_utilities.wald_protocol_v2(ms_1, pan_1, 4, sensor, n_bands)
        lms_2 = upsample(ms_2, size=pan_2.shape[-2:])
        pan_2_3c = pan_2.repeat(1, n_bands, 1, 1)

    data_dict = {
        'L0': {'ms': ms_0, 'lms': lms_0, 'pan': pan_0_3c},
        'L1': {'ms': ms_1, 'lms': lms_1, 'pan': pan_1_3c, 'gt': ms_0},
        'L2': {'ms': ms_2, 'lms': lms_2, 'pan': pan_2_3c, 'gt': ms_1, 'gt_enhanced': ms_0}
    }
    return data_dict, pan_0


def resolve_id_range(f):
    """决定要跑哪些 id：优先 --id；否则用 start/end/num"""
    total = f['lms'].shape[0]

    
    if args.id is not None:
        if not (0 <= args.id < total):
            raise ValueError(f"--id out of range: {args.id}, total={total}")
        return [args.id]

    
    start = args.start_id
    if start < 0:
        start = 0

    if args.end_id is not None and args.num is not None:
        
        end = args.end_id
    elif args.end_id is not None:
        end = args.end_id
    elif args.num is not None:
        end = start + args.num - 1
    else:
        end = total - 1

    # clamp
    end = min(end, total - 1)
    if start > end:
        return []

    return list(range(start, end + 1))



print("Init model...")
model = BF_NIR_conv_Cell(feat_dim=64, guide_dim=64,spa_edsr_num=4, mlp_dim=[256,128],NIR_dim=64, cell_decode=True).to(device)

print("Loading best model...")
model.load_state_dict(torch.load(args.save_pth_name, weights_only=True))
model.eval()


_ = get_cell(512, 512, device)


print("Loading data file...")
with h5py.File(args.path, 'r') as f:
    ids = resolve_id_range(f)
    print(f"Will run ids: {ids[:10]}{' ...' if len(ids) > 10 else ''} (count={len(ids)})")
    print(f"Sensor: {args.sensor}")

    for idx in ids:
        try:
            print(f"\n[Running] id={idx}")
            data_dict, pan_0 = build_data_dict_for_id(f, idx, args.sensor, n_bands=4)

            with torch.no_grad():
                final_out = model(
                    data_dict['L0']['pan'],
                    data_dict['L0']['lms'],
                    None,
                    target_shape=(int(512 * args.eval_scale), int(512 * args.eval_scale))
                )

            save_inference_results(
                final_out,
                data_dict['L0']['lms'],
                data_dict['L0']['ms'],
                pan_0,
                args.save_dir,
                idx,
                args.eval_scale,
                sensor=args.sensor
            )
        except Exception as e:
            print(f"[Error] id={idx} failed: {e}")
