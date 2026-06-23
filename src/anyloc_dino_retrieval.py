"""Lightweight AnyLoc-style DINOv2 retrieval over extracted drone frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torchvision import transforms
from tqdm import tqdm


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_dinov2(
    model_name: str,
    device: torch.device,
    dinov2_repo: Path,
    weights_path: Path | None,
) -> torch.nn.Module:
    torch_hub_dir = Path("outputs/torch_hub")
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))
    if weights_path is not None:
        model = torch.hub.load(
            str(dinov2_repo),
            model_name,
            source="local",
            weights=str(weights_path),
        )
    else:
        model = torch.hub.load(str(dinov2_repo), model_name, source="local")
    model.eval().to(device)
    return model


def preprocess_image(path: Path, max_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = transform(image)
    _, height, width = tensor.shape
    height = (height // 14) * 14
    width = (width // 14) * 14
    tensor = transforms.CenterCrop((height, width))(tensor)
    return tensor.unsqueeze(0)


def patch_descriptors_for_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    max_size: int,
) -> np.ndarray:
    tensor = preprocess_image(image_path, max_size).to(device)
    with torch.no_grad():
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    return patch_tokens.cpu().numpy().squeeze(0)


def mean_pool_descriptor(patch_descriptors: np.ndarray) -> np.ndarray:
    descriptor = patch_descriptors.mean(axis=0)
    norm = np.linalg.norm(descriptor)
    return descriptor / max(norm, 1e-12)


def gem_pool_descriptor(patch_descriptors: np.ndarray, gem_p: float) -> np.ndarray:
    # AnyLoc's GeM variant handles negative ViT features by applying the
    # signed real-valued equivalent of the complex root used in their scripts.
    powered_mean = np.mean(np.power(patch_descriptors, gem_p), axis=0)
    descriptor = np.sign(powered_mean) * np.power(np.abs(powered_mean), 1.0 / gem_p)
    norm = np.linalg.norm(descriptor)
    return descriptor / max(norm, 1e-12)


def fit_vlad_codebook(
    image_patch_descriptors: list[np.ndarray],
    num_clusters: int,
    max_train_patches: int,
    random_seed: int,
) -> MiniBatchKMeans:
    patches = np.vstack(image_patch_descriptors)
    if len(patches) > max_train_patches:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(patches), size=max_train_patches, replace=False)
        patches = patches[indices]
    print(f"VLAD training patches: {len(patches)}")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=4096,
        n_init="auto",
        random_state=random_seed,
    )
    kmeans.fit(patches)
    return kmeans


def vlad_descriptor(patch_descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    similarities = patch_descriptors @ centers.T
    labels = similarities.argmax(axis=1)
    residuals = np.zeros_like(centers, dtype=np.float32)
    for cluster_idx in range(len(centers)):
        members = patch_descriptors[labels == cluster_idx]
        if len(members) == 0:
            continue
        residual = members - centers[cluster_idx]
        residuals[cluster_idx] = residual.sum(axis=0)
    residuals = residuals.reshape(-1)
    norm = np.linalg.norm(residuals)
    return residuals / max(norm, 1e-12)


def compute_descriptors(
    rows: list[dict[str, str]],
    model_name: str,
    max_size: int,
    dinov2_repo: Path,
    weights_path: Path | None,
    aggregation: str,
    num_clusters: int,
    max_train_patches: int,
    random_seed: int,
    gem_p: float = 3.0,
) -> np.ndarray:
    device = choose_device()
    print(f"device: {device}")
    print(f"model: {model_name}")
    model = load_dinov2(model_name, device, dinov2_repo, weights_path)
    image_patch_descriptors = []
    for row in tqdm(rows, desc="DINOv2 descriptors"):
        image_patch_descriptors.append(
            patch_descriptors_for_image(
                model,
                Path(row["frame_path"]),
                device=device,
                max_size=max_size,
            )
        )
    if aggregation == "mean":
        return np.vstack([mean_pool_descriptor(desc) for desc in image_patch_descriptors])
    if aggregation == "gem":
        return np.vstack([gem_pool_descriptor(desc, gem_p) for desc in image_patch_descriptors])
    if aggregation == "vlad":
        codebook = fit_vlad_codebook(
            image_patch_descriptors,
            num_clusters=num_clusters,
            max_train_patches=max_train_patches,
            random_seed=random_seed,
        )
        centers = codebook.cluster_centers_.astype(np.float32)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        return np.vstack([vlad_descriptor(desc, centers) for desc in image_patch_descriptors])
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def local_xy_from_latlon(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    earth_radius_m = 6_378_137.0
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    return (
        (lon - lon0) * math.cos(lat0) * earth_radius_m,
        (lat - lat0) * earth_radius_m,
    )


def position_error_m(query: dict[str, str], reference: dict[str, str], origin: tuple[float, float]) -> float:
    qx, qy = local_xy_from_latlon(
        float(query["ground_latitude"]),
        float(query["ground_longitude"]),
        origin[0],
        origin[1],
    )
    rx, ry = local_xy_from_latlon(
        float(reference["ground_latitude"]),
        float(reference["ground_longitude"]),
        origin[0],
        origin[1],
    )
    return math.hypot(qx - rx, qy - ry)


def split_reference_query(rows: list[dict[str, str]], reference_stride: int) -> tuple[list[int], list[int]]:
    reference_indices = [idx for idx in range(len(rows)) if idx % reference_stride == 0]
    query_indices = [idx for idx in range(len(rows)) if idx % reference_stride != 0]
    return reference_indices, query_indices


def run_retrieval(
    rows: list[dict[str, str]],
    descriptors: np.ndarray,
    reference_stride: int,
    max_time_gap_s: float | None,
    top_k: int,
) -> list[dict[str, object]]:
    reference_indices, query_indices = split_reference_query(rows, reference_stride)
    reference_desc = descriptors[reference_indices]
    query_desc = descriptors[query_indices]
    similarities = query_desc @ reference_desc.T
    origin = (float(rows[0]["ground_latitude"]), float(rows[0]["ground_longitude"]))

    results = []
    for q_pos in range(len(query_indices)):
        query_index = query_indices[q_pos]
        query_time = float(rows[query_index]["start_seconds"])
        allowed = np.ones(len(reference_indices), dtype=bool)
        if max_time_gap_s is not None and max_time_gap_s > 0:
            allowed = np.array(
                [
                    abs(float(rows[ref_index]["start_seconds"]) - query_time) <= max_time_gap_s
                    for ref_index in reference_indices
                ]
            )
            if not allowed.any():
                allowed = np.ones(len(reference_indices), dtype=bool)
        masked_scores = similarities[q_pos].copy()
        masked_scores[~allowed] = -np.inf
        top_positions = np.argsort(masked_scores)[::-1][: max(1, top_k)]
        ref_pos = int(top_positions[0])
        reference_index = reference_indices[int(ref_pos)]
        query = rows[query_index]
        reference = rows[reference_index]
        top_errors = [
            position_error_m(query, rows[reference_indices[int(pos)]], origin)
            for pos in top_positions
            if np.isfinite(masked_scores[int(pos)])
        ]
        results.append(
            {
                "query_frame_count": int(query["frame_count"]),
                "query_frame_path": query["frame_path"],
                "reference_frame_count": int(reference["frame_count"]),
                "reference_frame_path": reference["frame_path"],
                "similarity": float(similarities[q_pos, ref_pos]),
                "time_gap_s": abs(
                    float(query["start_seconds"]) - float(reference["start_seconds"])
                ),
                "top_k": len(top_errors),
                "oracle_top_k_error_m": min(top_errors) if top_errors else math.nan,
                "query_ground_latitude": float(query["ground_latitude"]),
                "query_ground_longitude": float(query["ground_longitude"]),
                "estimated_ground_latitude": float(reference["ground_latitude"]),
                "estimated_ground_longitude": float(reference["ground_longitude"]),
                "position_error_m": position_error_m(query, reference, origin),
            }
        )
    return results


def write_results(results: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summarize(results: list[dict[str, object]]) -> dict[str, float | int]:
    errors = np.array([float(row["position_error_m"]) for row in results])
    oracle_errors = np.array([float(row["oracle_top_k_error_m"]) for row in results])
    return {
        "queries": len(results),
        "mean_error_m": float(errors.mean()) if len(errors) else 0.0,
        "median_error_m": float(np.median(errors)) if len(errors) else 0.0,
        "p90_error_m": float(np.percentile(errors, 90)) if len(errors) else 0.0,
        "max_error_m": float(errors.max()) if len(errors) else 0.0,
        "oracle_top_k_mean_error_m": float(oracle_errors.mean()) if len(oracle_errors) else 0.0,
        "oracle_top_k_median_error_m": float(np.median(oracle_errors)) if len(oracle_errors) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--descriptor-cache", type=Path, default=Path("outputs/anyloc/descriptors.npy"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/anyloc/summary.json"))
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--dinov2-repo", type=Path, default=Path("third_party/dinov2"))
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("outputs/models/dinov2/dinov2_vits14_pretrain.pth"),
    )
    parser.add_argument("--max-size", type=int, default=518)
    parser.add_argument("--aggregation", choices=["mean", "gem", "vlad"], default="mean")
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--max-train-patches", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--reference-stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-time-gap-s",
        type=float,
        default=0,
        help="Optional temporal retrieval window for recorded-flight simulation. 0 disables it.",
    )
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    args.descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.descriptor_cache.exists() and not args.recompute:
        descriptors = np.load(args.descriptor_cache)
    else:
        weights_path = args.weights_path if args.weights_path.exists() else None
        descriptors = compute_descriptors(
            rows,
            args.model_name,
            args.max_size,
            args.dinov2_repo,
            weights_path,
            args.aggregation,
            args.num_clusters,
            args.max_train_patches,
            args.random_seed,
            args.gem_p,
        )
        np.save(args.descriptor_cache, descriptors)

    max_time_gap_s = args.max_time_gap_s if args.max_time_gap_s > 0 else None
    results = run_retrieval(
        rows,
        descriptors,
        reference_stride=args.reference_stride,
        max_time_gap_s=max_time_gap_s,
        top_k=args.top_k,
    )
    write_results(results, args.output_csv)
    summary = summarize(results)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
