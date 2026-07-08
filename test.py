import argparse
import os

import h5py
import scipy.io as sio
import torch
import torch.nn.functional as F

from models.model_INR import BF_NIR_conv_Cell


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="H5 file path")
parser.add_argument("--id", type=int, default=None, help="single sample index (optional)")
parser.add_argument("--start_id", type=int, default=0, help="batch start index (inclusive)")
parser.add_argument("--end_id", type=int, default=None, help="batch end index (inclusive). If None, use --num or last index")
parser.add_argument("--num", type=int, default=20, help="number of samples to run from start_id (optional)")
parser.add_argument("--save_dir", type=str, default=".\\new", help="result save directory")
parser.add_argument("--sensor", type=str, default="WV3", help="sensor type saved in result metadata")
parser.add_argument("-n", "--eval_scale", type=float, default=1.0, help="quantitative evaluation scale relative to original PAN")
parser.add_argument(
    "--save_pth_name",
    type=str,
    default=r"temp_weights\temp_best_model_id0.pth",
    help="best model path for inference",
)
args = parser.parse_args()
if args.eval_scale <= 0:
    raise ValueError("--eval_scale/-n must be greater than 0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cell(H, W, device):
    hy = 2 / H
    hx = 2 / W
    cell = torch.tensor([hx, hy], dtype=torch.float32, device=device)
    return cell.unsqueeze(0).unsqueeze(0).repeat(1, H * W, 1)


def resize_for_eval(x, eval_scale):
    if eval_scale == 1:
        return x
    return F.interpolate(x, scale_factor=1 / eval_scale, mode="bicubic", align_corners=False)


def format_scale(eval_scale):
    if float(eval_scale).is_integer():
        return str(int(eval_scale))
    return f"{eval_scale:g}"


def to_numpy_HWC(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()


def save_inference_results(
    output_tensor,
    ref_tensor,
    ms_tensor,
    pan_tensor,
    save_dir,
    data_id,
    eval_scale,
    sensor,
    gt_tensor=None,
    lms_tensor=None,
    ms_eval_up_tensor=None,
    model_x1_up_tensor=None,
):
    os.makedirs(save_dir, exist_ok=True)

    save_dict = {
        "I_MS": to_numpy_HWC(ref_tensor),
        "proposed": to_numpy_HWC(output_tensor),
        "I_MS_LR": to_numpy_HWC(ms_tensor),
        "I_PAN": to_numpy_HWC(pan_tensor),
        "ratio": 4,
        "eval_scale": eval_scale,
        "sensor_type": sensor.upper(),
    }
    if gt_tensor is not None:
        save_dict["gt"] = to_numpy_HWC(gt_tensor)
        if lms_tensor is not None:
            save_dict["I_MS_UP"] = to_numpy_HWC(lms_tensor)
    if ms_eval_up_tensor is not None:
        save_dict["I_MS_eval_UP"] = to_numpy_HWC(ms_eval_up_tensor)
    if model_x1_up_tensor is not None:
        save_dict["proposed_x1_UP"] = to_numpy_HWC(model_x1_up_tensor)

    out_path = os.path.join(save_dir, f"{data_id}X{format_scale(eval_scale)}_best.mat")
    sio.savemat(out_path, save_dict)
    print(f"[Finished] id={data_id} saved: {out_path}")


def read_tensor(f, key, idx):
    return torch.from_numpy(f[key][idx]).unsqueeze(0).float().to(device)


def build_data_for_id(f, idx, eval_scale, n_bands=8):
    lms_full = read_tensor(f, "lms", idx)
    ms_full = read_tensor(f, "ms", idx)
    pan_full = torch.from_numpy(f["pan"][idx]).float().to(device)
    gt_full = None
    for gt_key in ("gt", "GT"):
        if gt_key in f:
            gt_full = read_tensor(f, gt_key, idx)
            break

    if pan_full.dim() == 2:
        pan_full = pan_full.unsqueeze(0).unsqueeze(0)
    elif pan_full.dim() == 3:
        pan_full = pan_full.unsqueeze(0)

    lms_eval = resize_for_eval(lms_full, eval_scale)
    ms_eval = resize_for_eval(ms_full, eval_scale)
    pan_eval = resize_for_eval(pan_full, eval_scale)
    pan_eval_3c = pan_eval.repeat(1, n_bands, 1, 1)

    data_dict = {
        "L0": {
            "ms": ms_eval,
            "lms": lms_eval,
            "pan": pan_eval_3c,
        }
    }
    originals = {
        "lms": lms_full,
        "ms": ms_full,
        "pan": pan_full,
        "gt": gt_full,
    }
    return data_dict, originals


def resolve_id_range(f):
    total = f["lms"].shape[0]

    if args.id is not None:
        if not (0 <= args.id < total):
            raise ValueError(f"--id out of range: {args.id}, total={total}")
        return [args.id]

    start = max(args.start_id, 0)
    if args.end_id is not None:
        end = args.end_id
    elif args.num is not None:
        end = start + args.num - 1
    else:
        end = total - 1

    end = min(end, total - 1)
    if start > end:
        return []

    return list(range(start, end + 1))


print("Init model...")
model = BF_NIR_conv_Cell(
    feat_dim=64,
    guide_dim=64,
    spa_edsr_num=6,
    mlp_dim=[256, 128],
    NIR_dim=48,
    cell_decode=True,
).to(device)

print("Loading best model...")
model.load_state_dict(torch.load(args.save_pth_name, weights_only=True))
model.eval()

print("Loading data file...")
with h5py.File(args.path, "r") as f:
    ids = resolve_id_range(f)
    print(f"Will run ids: {ids[:10]}{' ...' if len(ids) > 10 else ''} (count={len(ids)})")
    print(f"Sensor: {args.sensor}")
    print(f"Evaluation scale: {args.eval_scale} (inputs resized by 1/{args.eval_scale}; saved tensors keep original size)")

    for idx in ids:
        try:
            print(f"\n[Running] id={idx}")
            data_dict, originals = build_data_for_id(f, idx, args.eval_scale, n_bands=8)
            ref_full = originals["gt"] if originals["gt"] is not None else originals["lms"]
            final_shape = ref_full.shape[-2:]
            final_cell = get_cell(final_shape[0], final_shape[1], device)
            pan_shape = originals["pan"].shape[-2:]
            eval_shape = data_dict["L0"]["pan"].shape[-2:]
            eval_cell = get_cell(eval_shape[0], eval_shape[1], device)

            with torch.no_grad():
                final_out = model(
                    data_dict["L0"]["pan"],
                    data_dict["L0"]["lms"],
                    None,
                    cell=final_cell,
                    target_shape=final_shape,
                )
                ms_eval_up = F.interpolate(
                    data_dict["L0"]["ms"],
                    size=pan_shape,
                    mode="bicubic",
                    align_corners=False,
                )
                model_x1 = model(
                    data_dict["L0"]["pan"],
                    data_dict["L0"]["lms"],
                    None,
                    cell=eval_cell,
                    target_shape=eval_shape,
                )
                model_x1_up = F.interpolate(
                    model_x1,
                    size=pan_shape,
                    mode="bicubic",
                    align_corners=False,
                )

            save_inference_results(
                final_out,
                ref_full,
                originals["ms"],
                originals["pan"],
                args.save_dir,
                idx,
                args.eval_scale,
                sensor=args.sensor,
                gt_tensor=originals["gt"],
                lms_tensor=originals["lms"],
                ms_eval_up_tensor=ms_eval_up,
                model_x1_up_tensor=model_x1_up,
            )
        except Exception as e:
            print(f"[Error] id={idx} failed: {e}")
