import os
import time
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor, tensor2img

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def load_img(path):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def add_noise(img, sigma=25):
    noise = np.random.randn(*img.shape) * (sigma / 255.0)
    noisy = np.clip(img + noise, 0.0, 1.0)
    return noisy

def psnr(ref, img):
    ref = np.clip(ref, 0, 1)
    img = np.clip(img, 0, 1)
    return peak_signal_noise_ratio(ref, img, data_range=1.0)

def ssim(ref, img):
    return structural_similarity(ref, img, channel_axis=2, data_range=1.0)

def tensor_from_np(img):
    return img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0)

# -------------------------------------------------------------------
# Load NAFNet baseline
# -------------------------------------------------------------------

def load_nafnet(config_path="models/NAFNet-width64.yml"):
    opt = parse(config_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)
    return model

# -------------------------------------------------------------------
# Evaluate a single model
# -------------------------------------------------------------------

def evaluate_model(model, clean_imgs, sigma=25):
    psnr_list = []
    ssim_list = []

    # Measure latency
    torch.cuda.synchronize()
    start = time.time()

    for clean in tqdm(clean_imgs, desc="Evaluating"):
        noisy = add_noise(clean, sigma=sigma)

        # Convert to tensor
        noisy_t = tensor_from_np(noisy).cuda()

        model.feed_data({'lq': noisy_t})
        model.test()
        out = model.get_current_visuals()['result']
        out = tensor2img([out]) / 255.0  # convert to [0,1]

        psnr_list.append(psnr(clean, out))
        ssim_list.append(ssim(clean, out))

    torch.cuda.synchronize()
    total_time = time.time() - start
    latency = total_time / len(clean_imgs)

    return np.mean(psnr_list), np.mean(ssim_list), latency

# -------------------------------------------------------------------
# Count parameters
# -------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------------------------------------------
# Main Evaluation Script
# -------------------------------------------------------------------

def main():

    print("\nLoading baseline NAFNet...")
    baseline = load_nafnet()
    baseline.net_g = baseline.net_g.cuda()
    baseline_name = "baseline"

    # Load pruned/quantized models ---------------------------------------------------
    models = {
        "baseline": baseline.net_g,
    }

    print("Loading pruned / quantized models...")

    # You will need to save your pruned models as .pth files in /models/
    # Example:
    # models["pruned30"] = torch.load("models/pruned30.pth")
    # models["pruned50"] = torch.load("models/pruned50.pth")

    # For now, we assume only baseline is available
    # You will add pruning/quantization versions here later

    # ------------------------------------------------------------------
    # Load dataset (DIV2K patches)
    # ------------------------------------------------------------------

    div2k_dir = "/content/drive/MyDrive/Machine Learning/DIV2K_patches"
    files = sorted([os.path.join(div2k_dir, f) for f in os.listdir(div2k_dir) if f.endswith(".png")])

    # Evaluate on first N images
    N = 50
    files = files[:N]
    print(f"\nLoaded {len(files)} evaluation patches.")

    clean_imgs = [load_img(f) for f in files]

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------

    results = []

    for name, net in models.items():
        print(f"\nEvaluating model: {name}")
        net.eval().cuda()

        p, s, latency = evaluate_model(baseline, clean_imgs)

        # model size
        tmp_path = f"{name}_tmp.pth"
        torch.save(net.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024**2)
        os.remove(tmp_path)

        params = count_params(net)

        print(f"\n{name} -> PSNR: {p:.2f}, SSIM: {s:.4f}, latency: {latency*1000:.2f} ms")
        print(f"params: {params}, size: {size_mb:.2f} MB\n")

        results.append([name, p, s, latency, params, size_mb])

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    with open("div2k_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PSNR", "SSIM", "Latency_s", "Params", "Size_MB"])
        writer.writerows(results)

    print("\nSaved results to div2k_results.csv")

# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
