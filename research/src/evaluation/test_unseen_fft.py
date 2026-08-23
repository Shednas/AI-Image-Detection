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


# Re-encode image as JPEG at the given quality level to simulate compression artefacts
def apply_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# The models were trained on images that process_data.py had already rewritten on
# disk: converted to RGB, resized to 256x256 with LANCZOS, and saved as JPEG q95.
# This reproduces that in memory so an unseen set can be scored through the same
# pipeline. data/unseen/ is never written to; process_data.py deletes files it
# cannot read, and that download is not reproducible cheaply.
def apply_training_preprocessing(img, target_size=256):
    if img.mode != "RGB":
        img = img.convert("RGB")
    resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Run sigmoid inference over a list of image paths and return probs + labels
def run_inference(model, paths, label, quality, device, match_training=False):
    probs, labels, names = [], [], []
    model.eval()
    with torch.no_grad():
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                # before any degradation, so degradation acts on the same input it
                # would have acted on during training
                if match_training:
                    img = apply_training_preprocessing(img)
                if quality:
                    img = apply_jpeg(img, quality)
                tensor = TRANSFORM(img).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(tensor).squeeze()).item()
                probs.append(prob)
                labels.append(label)
                names.append(p.name)
            except Exception as e:
                print(f"  Skipped {p.name}: {e}")
    return probs, labels, names


# Compute classification metrics over combined real+AI paths
def evaluate(model, real_paths, ai_paths, quality, device, match_training=False):
    rp, rl, rn = run_inference(model, real_paths, label=1, quality=quality, device=device,
                               match_training=match_training)
    ap, al, an = run_inference(model, ai_paths, label=0, quality=quality, device=device,
                               match_training=match_training)
    all_p = rp + ap
    all_l = rl + al
    # returned beside the metrics so a threshold sweep is a laptop job later
    records = [{"file": n, "label": l, "p_real": p}
               for n, l, p in zip(rn + an, all_l, all_p)]
    if not all_p:
        return None, records
    metrics = compute_metrics(np.array(all_l), np.array(all_p))
    print(f"    Acc: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  AUC: {metrics['roc_auc']:.4f}")
    return metrics, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FFT on unseen datasets")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--dataset", type=str, default="chameleon", choices=["chameleon", "mnw", "all"])
    parser.add_argument("--degradation", type=str, default="none", choices=["none", "light", "heavy", "all"])
    parser.add_argument("--n_samples", type=int, default=200, help="Images per class to sample")
    parser.add_argument("--match_training_preprocessing", action="store_true",
                        help="Resize with LANCZOS and re-encode as JPEG q95 first, "
                             "reproducing what process_data.py did to the training set")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Where to write results, default results/MODEL/stage_N/unseen")
    parser.add_argument("--dump_probabilities", action="store_true",
                        help="Also write the raw per-image probability and label, so a "
                             "threshold sweep needs no second GPU run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(f"checkpoints/fft/stage_{args.stage}/best_fft.pt")

    model = FFTDetector(image_size=256, num_bands=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded FFT Stage {args.stage} from {ckpt}")

    results_dir = Path(args.results_dir or f"results/fft/stage_{args.stage}/unseen")
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['chameleon', 'mnw'] if args.dataset == 'all' else [args.dataset]
    degradations = ['none', 'light', 'heavy'] if args.degradation == 'all' else [args.degradation]

    for ds in datasets:
        roots = DATASET_ROOTS[ds]
        real_paths = collect_paths(roots['real'], args.n_samples) if roots.get('real') else []
        ai_paths = collect_paths(roots['ai'], args.n_samples) if roots.get('ai') else []
        print(f"\nDataset: {ds}  |  Real: {len(real_paths)}  AI: {len(ai_paths)}")

        out = {"dataset": ds, "stage": args.stage, "model": "fft",
               "match_training_preprocessing": args.match_training_preprocessing,
               "n_real": len(real_paths), "n_ai": len(ai_paths), "results": {}}

        raw = {}
        for deg in degradations:
            print(f"  Degradation: {deg}")
            m, records = evaluate(model, real_paths, ai_paths, JPEG_QUAL[deg], device,
                                  match_training=args.match_training_preprocessing)
            if m:
                out["results"][deg] = m
            raw[deg] = records

        save_path = results_dir / f"unseen_{ds}.json"
        with open(save_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {save_path}")

        if args.dump_probabilities:
            dump = {"dataset": ds, "stage": args.stage, "model": "fft",
                    "match_training_preprocessing": args.match_training_preprocessing,
                    "note": "p_real is the raw sigmoid output. Training used {ai_generated: 0, real: 1}, so P(AI) = 1 - p_real, and label uses the same mapping.",
                    "results": raw}
            dump_path = results_dir / f"unseen_{ds}_probabilities.json"
            with open(dump_path, 'w') as f:
                json.dump(dump, f, indent=2)
            print(f"  Saved: {dump_path}")
