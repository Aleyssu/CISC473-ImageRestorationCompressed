import os
import time
import torch
import cv2
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img

#############################
#     Helper Functions
#############################

def imread(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def img2tensor(img):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=False, float32=True)

def run_inference(model, img_np):
    """Run NAFNet inference and return restored image (numpy)."""
    inp = img2tensor(img_np)
    model.feed_data({"lq": inp.unsqueeze(0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    out = tensor2img([visuals["result"]])
    return out


#############################
#   Evaluation Functions
#############################

def evaluate_folder(model, clean_dir, noisy_dir):
    psnrs, ssims = [], []
    files = sorted(os.listdir(clean_dir))

    for fname in files:
        clean_path = os.path.join(clean_dir, fname)
        noisy_path = os.path.join(noisy_dir, fname)

        clean = imread(clean_path)
        noisy = imread(noisy_path)

        restored = run_inference(model, noisy)

        p = psnr(clean, restored, data_range=255)
        s = ssim(clean, restored, channel_axis=2, data_range=255)

        psnrs.append(p)
        ssims.append(s)

    return np.mean(psnrs), np.mean(ssims)


def model_size_mb(model_path):
    return os.path.getsize(model_path) / (1024**2)


def measure_speed(model, sample_img, runs=10):
    """Measure average inference time in milliseconds."""
    # warmup
    for _ in range(3):
        run_inference(model, sample_img)

    start = time.perf_counter()
    for _ in range(runs):
        run_inference(model, sample_img)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / runs
    return avg_ms


#############################
#     Main Evaluation
#############################

def load_model(config_path, weights_path=None):
    """Load a NAFNet model using Basicsr parsing."""
    opt = parse(config_path, is_train=False)
    opt["dist"] = False
    model = create_model(opt)

    if weights_path:
        model.net_g.load_state_dict(torch.load(weights_path), strict=True)

    return model


def evaluate_all_models():
    print("🔍 Starting Evaluation...")

    CLEAN_DIR = "data/clean"
    NOISY_DIR = "data/noisy"

    CONFIG = "models/NAFNet-width64.yml"
    SAMPLE_NOISY_IMG = imread(os.path.join(NOISY_DIR, os.listdir(NOISY_DIR)[0]))

    models = {
        "baseline": "weights/nafnet_baseline.pth",
        "pruned30": "weights/nafnet_pruned30.pth",
        "pruned50": "weights/nafnet_pruned50.pth",
        "quantized": "weights/nafnet_quantized.pth",
    }

    results = {}

    for name, path in models.items():
        print(f"\n📌 Evaluating: {name}")

        model = load_model(CONFIG, path)

        # Mean PSNR/SSIM
        mean_psnr, mean_ssim = evaluate_folder(model, CLEAN_DIR, NOISY_DIR)

        # Inference speed
        speed = measure_speed(model, SAMPLE_NOISY_IMG)

        # Model size
        size = model_size_mb(path)

        results[name] = {
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "speed_ms": speed,
            "size_mb": size
        }

        print(f"   PSNR:   {mean_psnr:.3f} dB")
        print(f"   SSIM:   {mean_ssim:.4f}")
        print(f"   Speed:  {speed:.2f} ms/img")
        print(f"   Size:   {size:.2f} MB")

    print("\n\n🎉 Finished Evaluation.")
    return results


if __name__ == "__main__":
    results = evaluate_all_models()
    print("\n📊 Final Summary:")
    print(results)
