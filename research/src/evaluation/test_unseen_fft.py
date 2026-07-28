import argparse
import json
import io
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.models.fft_model import FFTDetector
from src.evaluation.metrics import compute_metrics

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
JPEG_QUAL = {'none': None, 'light': 75, 'heavy': 25}

DATASET_ROOTS = {
    'chameleon': {'real': 'data/unseen/chameleon/real', 'ai': 'data/unseen/chameleon/fake'},
    'mnw': {'real': None, 'ai': 'data/unseen/mnw'},
}


# Recursively gather image paths from a folder, sampling up to n with a fixed seed
def collect_paths(folder, n, seed=42):
    folder = Path(folder)
    if not folder.exists():
        print(f"  WARNING: {folder} not found")
        return []
    paths = [p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS]
    random.seed(seed)
    return random.sample(paths, min(n, len(paths)))


# Re-encode image as JPEG at the given quality level to simulate compression artifacts
def apply_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# Run sigmoid inference over a list of image paths and return probs + labels
def run_inference(model, paths, label, quality, device):
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                if quality:
                    img = apply_jpeg(img, quality)
                tensor = TRANSFORM(img).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(tensor).squeeze()).item()
                probs.append(prob)
                labels.append(label)
            except Exception as e:
                print(f"  Skipped {p.name}: {e}")
    return probs, labels


# Compute classification metrics over combined real+AI paths
def evaluate(model, real_paths, ai_paths, quality, device):
    rp, rl = run_inference(model, real_paths, label=1, quality=quality, device=device)
    ap, al = run_inference(model, ai_paths, label=0, quality=quality, device=device)
    all_p = rp + ap
    all_l = rl + al
    if not all_p:
        return None
    metrics = compute_metrics(np.array(all_l), np.array(all_p))
    print(f"    Acc: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  AUC: {metrics['roc_auc']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FFT on unseen datasets")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--dataset", type=str, default="chameleon", choices=["chameleon", "mnw", "all"])
    parser.add_argument("--degradation", type=str, default="none", choices=["none", "light", "heavy", "all"])
    parser.add_argument("--n_samples", type=int, default=200, help="Images per class to sample")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(f"checkpoints/fft/stage_{args.stage}/best_fft.pt")

    model = FFTDetector(image_size=256, num_bands=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded FFT Stage {args.stage} from {ckpt}")

    results_dir = Path(f"results/fft/stage_{args.stage}/unseen")
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['chameleon', 'mnw'] if args.dataset == 'all' else [args.dataset]
    degradations = ['none', 'light', 'heavy'] if args.degradation == 'all' else [args.degradation]

    for ds in datasets:
        roots = DATASET_ROOTS[ds]
        real_paths = collect_paths(roots['real'], args.n_samples) if roots.get('real') else []
        ai_paths = collect_paths(roots['ai'], args.n_samples) if roots.get('ai') else []
        print(f"\nDataset: {ds}  |  Real: {len(real_paths)}  AI: {len(ai_paths)}")

        out = {"dataset": ds, "stage": args.stage, "model": "fft",
               "n_real": len(real_paths), "n_ai": len(ai_paths), "results": {}}

        for deg in degradations:
            print(f"  Degradation: {deg}")
            m = evaluate(model, real_paths, ai_paths, JPEG_QUAL[deg], device)
            if m:
                out["results"][deg] = m

        save_path = results_dir / f"unseen_{ds}.json"
        with open(save_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {save_path}")
