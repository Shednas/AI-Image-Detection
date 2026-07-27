import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


MODEL_EXPLANATIONS = {
    "Spatial_CNN": "The CNN model analyses spatial structure using deep convolutional features from a ResNet-50 backbone trained on ImageNet.",
    "Frequency_FFT": "The FFT model analyses frequency-domain patterns in the image spectrum, detecting artefacts that AI generation introduces into specific frequency bands.",
    "Hybrid_Fusion": "The Hybrid model combines both spatial (CNN) and frequency (FFT) features, giving it a broader view of image authenticity than either branch alone.",
    "Handcrafted_STM": "The STM model uses handcrafted features (HOG, LBP, DCT, colour statistics, and noise residual) with a LightGBM classifier. No neural network is involved.",
}

VERDICT_EXPLANATIONS = {
    "AI_GENERATED": "The model classified this image as AI-generated. The confidence score reflects how strongly the features pointed toward synthetic origin.",
    "AUTHENTIC": "The model classified this image as authentic. This may be a photograph, screenshot, digital art, scan, or any non-AI-generated image. The confidence score reflects how strongly the features pointed toward authentic, non-AI origin.",
}

FEATURE_GROUP_SLICES = {
    "HOG (Edge shapes)": slice(0, 1764),
    "LBP (Micro-textures)": slice(1764, 1792),
    "DCT (Block patterns)": slice(1792, 1798),
    "Color Stats (Distributions)": slice(1798, 1810),
    "Noise Residual (Sensor noise)": slice(1810, 1822),
}


# map continuous P(AI) to a human-readable confidence zone label
def _probability_zone(p_ai: float) -> dict:
    if p_ai < 0.2:
        return {"key": "very_likely_real", "label": "Very likely authentic"}
    if p_ai < 0.4:
        return {"key": "likely_real", "label": "Likely authentic"}
    if p_ai < 0.6:
        return {"key": "uncertain", "label": "Borderline - uncertain"}
    if p_ai < 0.8:
        return {"key": "likely_ai", "label": "Likely AI-generated"}
    return {"key": "very_likely_ai", "label": "Very likely AI-generated"}


class ResultsHandler:
    # assemble the full API response for a single-image analysis
    def format_single(self, raw_output, image_bytes, model_name, image_tensor=None, model=None):
        p_real = raw_output["p_real"]
        p_ai = raw_output["p_ai"] # computed once, in the pipeline

        return {
            "model_name": raw_output["model_name"],
            "verdict": raw_output["verdict"],
            "latency_ms": raw_output["latency_ms"],
            "p_real": p_real,
            "ai_pct": round(p_ai * 100, 1),
            "confidence_pct": round(max(p_real, p_ai) * 100, 1),
            "zone": _probability_zone(p_ai),
            "model_explanation": MODEL_EXPLANATIONS.get(raw_output["model_name"], ""),
            "verdict_explanation": VERDICT_EXPLANATIONS.get(raw_output["verdict"], ""),
            "visualizations": self._build_visualizations(image_bytes, raw_output, model_name, image_tensor, model),
        }

    # collapse raw row results into aggregate stats
    def format_batch_summary(self, results):
        valid = [r for r in results if "error" not in r]
        ai_count = sum(1 for r in valid if r["verdict"] == "AI_GENERATED")
        real_count = len(valid) - ai_count
        total_valid = len(valid) or 1
        return {
            "total": len(results),
            "valid": len(valid),
            "ai_count": ai_count,
            "real_count": real_count,
            "ai_pct": round(ai_count / total_valid * 100, 1),
            "real_pct": round(real_count / total_valid * 100, 1),
            "rows": results,
        }

    # only generate the viz types that the chosen model actually supports
    def _build_visualizations(self, image_bytes, raw_output, model_name, image_tensor, model):
        viz = {
            "rgb_distribution": self.generate_rgb_distribution(image_bytes),
            "generic_metrics": self.generate_generic_metrics(image_bytes),
        }

        if model_name in ("cnn", "hybrid"):
            heatmap_data = None
            if model is not None and image_tensor is not None:
                # hybrid holds its ResNet as cnn_branch, cnn wraps it as backbone
                target_layer = model.backbone.layer4 if model_name == "cnn" else model.cnn_branch.layer4
                heatmap_data = self._safe_gradcam(
                    image_tensor, model, target_layer, image_bytes, raw_output["verdict"]
                )

            description = (
                "Grad-CAM heatmap highlights the regions that pushed the model toward this "
                "particular verdict. Red/warm areas contributed most strongly; blue/cool areas "
                "had little influence."
            )
            if model_name == "hybrid":
                description += (
                    " It is taken from the spatial branch, so it explains that branch rather than "
                    "the fused decision. The frequency spectrogram covers the other branch."
                )

            viz["heatmap"] = {"data": heatmap_data, "description": description}

        if model_name in ("fft", "hybrid"):
            viz["spectrogram"] = {
                "data": self.generate_frequency_map(image_bytes),
                "description": (
                    "Frequency spectrogram shows the energy distribution across the image spectrum (DC at centre). "
                    "AI generators often introduce periodic spikes or unnatural energy concentrations in mid-to-high frequency bands."
                ),
            }

        if model_name == "stm":
            viz["feature_importance"] = {
                "data": self._safe_feature_importance(model, image_tensor) if model is not None else None,
                "description": (
                    "Feature contribution shows how much each handcrafted feature group moved the LightGBM "
                    "prediction for this image, measured as per-prediction SHAP values. Higher percentage "
                    "means that group carried more weight in this particular verdict."
                ),
            }

        return viz

    # smoothness ratio flags AI over-smoothing of channel distributions
    def generate_rgb_distribution(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        bins = list(range(0, 257, 8))
        r_hist, _ = np.histogram(arr[:, :, 0].ravel(), bins=bins)
        g_hist, _ = np.histogram(arr[:, :, 1].ravel(), bins=bins)
        b_hist, _ = np.histogram(arr[:, :, 2].ravel(), bins=bins)
        combined_hist = np.concatenate([r_hist, g_hist, b_hist]).astype(float)
        smoothness = round(float(combined_hist.std() / (combined_hist.mean() + 1e-8)), 2)
        return {
            "labels": list(range(0, 256, 8)),
            "r": r_hist.tolist(),
            "g": g_hist.tolist(),
            "b": b_hist.tolist(),
            "smoothness": smoothness,
        }

    # gaussian residual is a crude proxy for sensor PRNU noise
    def generate_generic_metrics(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        contrast = round(float(gray.std() * 100), 1)
        residual = gray - gaussian_filter(gray, sigma=3)
        noise = round(float(residual.std() * 100), 1)
        return {"contrast": contrast, "noise": noise}

    # hanning window reduces spectral leakage at image borders
    def generate_frequency_map(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float32) / 255.0
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        window = np.outer(np.hanning(256), np.hanning(256))
        fft_shift = np.fft.fftshift(np.fft.fft2(gray * window))
        magnitude = np.log1p(np.abs(fft_shift))
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        colored = (plt.get_cmap('plasma')(magnitude)[:, :, :3] * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(colored).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # feature extraction and the LightGBM call can both raise; a failed chart
    # should not cost the caller the verdict it came for
    def _safe_feature_importance(self, model, image_tensor):
        try:
            return self.generate_feature_importance(model, image_tensor)
        except Exception as e:
            print(f"Feature contribution failed: {e}")
            return None

    # per-prediction SHAP contributions, not feature_importances_. The latter is
    # a property of the trained model, so it returned an identical breakdown for
    # every image while the caption claimed it explained this one.
    def generate_feature_importance(self, model, image_tensor):
        if image_tensor is None:
            return None
        features = model.extract_features(image_tensor)
        contributions = model.lgbm_model.predict(features, pred_contrib=True)
        # last column is the bias term, which belongs to no feature group.
        # absolute values because a group that argues against the verdict still
        # influenced it, and signed sums would cancel within a group
        per_feature = np.abs(contributions[0, :-1])
        groups = {name: float(per_feature[sl].sum()) for name, sl in FEATURE_GROUP_SLICES.items()}
        total = sum(groups.values()) or 1.0
        return {k: round(v / total * 100, 1) for k, v in groups.items()}

    def generate_explanation(self, model_name, verdict, metrics):
        return f"{MODEL_EXPLANATIONS.get(model_name, '')} {VERDICT_EXPLANATIONS.get(verdict, '')}"

    # Grad-CAM can fail on some model/input combos; return None rather than 500
    def _safe_gradcam(self, image_tensor, model, target_layer, image_bytes, verdict):
        try:
            return self._generate_gradcam(image_tensor, model, target_layer, image_bytes, verdict)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            return None

    # full backward pass through the target layer; hooks are always cleaned up in finally
    def _generate_gradcam(self, image_tensor, model, target_layer, image_bytes, verdict):
        activations = {}
        gradients = {}
        h_fwd = target_layer.register_forward_hook(lambda m, i, o: activations.update({'a': o}))
        h_bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.update({'g': go[0]}))
        try:
            model.eval()
            tensor = image_tensor.clone().detach()
            with torch.enable_grad():
                output = model(tensor)
                model.zero_grad()
                # the single logit rises toward "real" under the {ai: 0, real: 1}
                # training mapping, so ascending it explains an AUTHENTIC verdict.
                # an AI verdict lives in the opposite direction and needs the
                # target negated, otherwise the heatmap argues the other case
                target = output.mean()
                if verdict == "AI_GENERATED":
                    target = -target
                target.backward()
            act = activations['a'].squeeze(0)
            grd = gradients['g'].squeeze(0)
            weights = grd.mean(dim=(1, 2))
            cam = (weights[:, None, None] * act).sum(dim=0)
            cam = torch.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_np = cam.detach().cpu().numpy()
        finally:
            h_fwd.remove()
            h_bwd.remove()
            model.zero_grad()
        cam_256 = np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        heatmap = (plt.get_cmap('jet')(cam_256)[:, :, :3] * 255).astype(np.uint8)
        orig = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256)))
        blended = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
