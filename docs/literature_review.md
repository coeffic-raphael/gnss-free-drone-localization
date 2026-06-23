# Literature Review

## Scope

The assignment asks for recent algorithms and tools for low-altitude drone visual navigation (20–200 m), with emphasis on open-source "paper with code" methods. The core problem is GNSS-denied localization: given a preprocessed reference flight with known positions, estimate the camera-center coordinate of each new query frame without using GNSS at inference time.

The most natural framing for this problem is **visual place recognition (VPR)**: build a database of reference images with known ground-projected coordinates, then retrieve the closest visual match for each query image. This literature review covers the VPR methods that are most relevant to low-altitude drone navigation, starting from the foundational work and arriving at the approach used in this project.

---

## 1. NetVLAD — Foundational Learning-Based VPR

**Paper:** Arandjelovic et al., "NetVLAD: CNN Features for Image Retrieval," CVPR 2016.
**Code:** https://github.com/Relja/netvlad

NetVLAD is the baseline against which all modern VPR methods are measured. It introduced end-to-end learning of a VLAD-style descriptor for place recognition, trained with weakly supervised contrastive learning on the Pittsburgh-250k dataset (urban driving imagery). The key contribution is the differentiable VLAD layer that aggregates local CNN features into a compact global descriptor.

NetVLAD established the dominant paradigm for the following decade: train a model on large-scale GPS-tagged urban data, then retrieve nearest neighbors at test time. This approach works well in urban environments similar to the training set, but degrades sharply when the test environment is out-of-distribution.

In the AnyLoc benchmark (Table IV), NetVLAD achieves only 19.7% R@1 on Nardo-Air (a GNSS-denied aerial drone dataset at 50 m altitude) — compared to 76.1% for AnyLoc-VLAD-DINOv2. This directly motivates the training-free approach used in this project: our reference flight is a single campus, not a large urban dataset, so training a NetVLAD-style model on it would overfit and not generalize.

---

## 2. MixVPR and CosPlace — Current Supervised SOTA

**MixVPR:** Ali-bey et al., "MixVPR: Feature Mixing for Visual Place Recognition," WACV 2023.
**Code:** https://github.com/amaralibey/MixVPR

**CosPlace:** Berton et al., "Rethinking Visual Geo-Localization for Large-Scale Applications," CVPR 2022.
**Code:** https://github.com/gmberton/CosPlace

MixVPR and CosPlace represent the state of the art in supervised VPR. MixVPR uses an MLP-based feature mixer trained on the GSV-Cities dataset (530,000 images, 62,000 places worldwide). CosPlace uses classification-based learning on the San Francisco XL dataset (40 million images with GPS and heading).

Both methods achieve strong performance on urban benchmarks: MixVPR reaches 83.9% average R@1 across structured environments in the AnyLoc evaluation. However, their performance collapses on aerial and unstructured environments: CosPlace achieves **0% R@1** on Nardo-Air (all queries incorrectly matched to a handful of visually similar fields), and MixVPR reaches only 32.4%. This failure demonstrates that large-scale urban training does not transfer to low-altitude drone imagery.

For this project, supervised methods are additionally inappropriate because the reference and test flights cover the same campus area: training on the reference flight would risk overfitting to the evaluation scene.

---

## 3. AnyLoc — Main Method

**Paper:** Keetha et al., "AnyLoc: Towards Universal Visual Place Recognition," RA-L / ICRA 2024.
**Code:** https://github.com/AnyLoc/AnyLoc
**PDF:** https://anyloc.github.io/assets/AnyLoc.pdf

AnyLoc is the central paper of this project. It proposes a universal, training-free VPR system based on frozen foundation model features, and evaluates it across 12 diverse datasets including aerial drone environments.

### Core pipeline

AnyLoc answers four design questions sequentially:

**A. Which foundation model?** Among joint-embedding methods (DINO, DINOv2), contrastive methods (CLIP), and masked autoencoders (MAE), DINOv2 provides the best features for VPR. MAE, which only has token-level supervision, performs the worst. DINO and DINOv2 use global image-level self-supervision that captures long-range patterns essential for place discrimination.

**B. Which features to extract?** Rather than using the CLS token (one vector per image), AnyLoc extracts per-pixel patch tokens from intermediate ViT layers. The paper shows that the **value facet of layer 31** of DINOv2 ViT-G14 gives the sharpest contrast in similarity maps, which is critical for discriminating visually similar places. Earlier layers have a high positional encoding bias; deeper value facets have the best spatial discriminability.

**C. How to aggregate?** AnyLoc compares global average pooling (GAP), generalized mean pooling (GeM), and soft/hard-assignment VLAD. The results are decisive (Table VII of the paper):

| Method | Baidu R@1 | Oxford R@1 | Descriptor size |
| --- | ---: | ---: | ---: |
| GAP (= mean pooling) | 41.6% | 78.5% | 1536 |
| GeM | 50.1% | 92.2% | 1536 |
| VLAD (hard) | 71.5% | 94.8% | 49152 |

Hard-assignment VLAD outperforms mean pooling by 30% on indoor data. This is the main quantitative gap between what this project implements and the full AnyLoc recommendation.

**D. Vocabulary construction?** For VLAD, the cluster vocabulary should be domain-specific. For aerial data specifically, using an aerial-domain vocabulary improves R@1 by 19% over a global vocabulary (Table V of the paper).

### Aerial results

The Nardo-Air dataset in AnyLoc closely mirrors our problem: GNSS-denied drone localization where query images are matched against a reference map. AnyLoc-VLAD-DINOv2 achieves 76.1% R@1 on Nardo-Air, compared to 0% for CosPlace and 32.4% for MixVPR. This result is the direct justification for using frozen DINOv2 features in this project.

### What this project uses vs. full AnyLoc

| Component | AnyLoc recommendation | This project |
| --- | --- | --- |
| Backbone | DINOv2 ViT-G14 | DINOv2 ViT-S14 (lighter) |
| Layer / facet | Layer 31, value facet | Mean over all patch tokens |
| Aggregation | VLAD (hard, 32 clusters) | Mean pooling |
| Vocabulary | Domain-specific (aerial) | Not applicable (no VLAD) |

Mean pooling was retained because VLAD aggregation, while improving raw retrieval candidates, did not improve the final temporally selected trajectory in our experiments. This likely reflects an interaction between vocabulary quality and the temporal reranking stage, and is noted as a future improvement.

---

## 4. DINOv2

**Paper:** Oquab et al., "DINOv2: Learning Robust Visual Features Without Supervision," 2023.
**Code:** https://github.com/facebookresearch/dinov2

DINOv2 is the frozen visual backbone used in this project. It trains a Vision Transformer on a large curated dataset (LVD-142M) with joint image-level and token-level self-supervised losses. The key property for VPR is that DINOv2 patch tokens encode rich semantic and spatial information without any task-specific supervision.

In our pipeline, DINOv2 patch tokens are extracted from each frame and mean-pooled into a single global descriptor. This descriptor serves as the visual signature for large-scale retrieval. Because DINOv2 is frozen and requires no training on the target data, the reference videos function as a pure database rather than a training set.

---

## 5. SuperPoint

**Paper:** DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description," CVPRW 2018.
**Code (via LightGlue):** https://github.com/cvg/LightGlue

SuperPoint is a self-supervised keypoint detector and descriptor trained on homographic pairs of synthetic and real images. It detects repeatable interest points and computes compact descriptors that are robust to viewpoint and illumination changes.

In this project, SuperPoint is used as the keypoint frontend for LightGlue local verification. It is also used in the frame-to-frame dead reckoning experiment to estimate pixel displacement between consecutive query frames.

---

## 6. LightGlue

**Paper:** Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed," ICCV 2023.
**Code:** https://github.com/cvg/LightGlue

LightGlue is a Transformer-based local feature matcher that pairs SuperPoint keypoints between two images. It is a lighter and faster successor to SuperGlue, with adaptive depth that terminates early when matches are sufficiently confident.

In this project, LightGlue verifies the small top-k candidate list returned by DINOv2 global retrieval. For each query frame, DINOv2 retrieves 6–10 reference candidates; LightGlue scores each pair by number of matches, RANSAC inliers, and inlier ratio. This local geometric verification step reduces the mean position error from 27.28 m (DINOv2 alone) to 19.15 m.

The hierarchical structure of our pipeline — global retrieval followed by local verification — mirrors the HLoc framework (Sarlin et al., CVPR 2019), which established this two-stage approach as the standard for large-scale visual localization.

---

## 7. Temporal Sequence Localization

The assignment is about a continuous flight, not isolated frames. A sequence-aware localization method significantly outperforms independent per-frame retrieval because consecutive frames should not jump randomly across the map.

The approach used in this project is a **Viterbi path selector** over the DINOv2 + LightGlue candidate lists:

- The visual score rewards high DINOv2 similarity and strong LightGlue matching.
- The transition score penalizes jumps larger than the expected drone speed.
- The Viterbi algorithm selects the minimum-cost consistent path through the candidate graph.

This is conceptually related to **SeqSLAM** (Milford and Wyeth, ICRA 2012), which showed that enforcing sequential consistency over a sliding window dramatically improves localization recall even with weak per-frame descriptors. The key difference is that our method works with a sparse top-k candidate graph rather than a dense similarity matrix.

Adding the Viterbi step reduces the mean error from 19.15 m (LightGlue reranking) to 18.83 m, with the main gain coming from reducing maximum errors (72.53 m vs 180.52 m for DINOv2 alone). A subsequent **Gaussian path smoothing** step (window w=19 frames, σ=5.4) then averages each estimated position with its temporal neighbours, pulling isolated wrong retrievals toward the correct neighbourhood. This reduces the mean error further to **14.16 m** and the maximum error from 72.53 m to 38.94 m.

---

## 8. Why We Did Not Train a Model

Training a model on the reference flights would be conceptually circular: the same area is used for both building the reference database and evaluating the query. AnyLoc explicitly motivates training-free VPR to avoid this problem. A frozen model makes the evaluation clean: the reference flight is a map, not a training set.

Additionally, the failure of CosPlace (0% on Nardo-Air) and the poor performance of MixVPR on aerial data confirm that urban-trained supervised methods cannot be applied out-of-the-box to low-altitude drone imagery without domain adaptation — which would require drone-specific labeled data that is not available here.

---

## 9. Satellite Image Matching for GNSS-Free Localization

### WildNav

**Paper:** Gurgu et al., "WildNav: Using Freely-Available Satellite Imagery for Wilderness UAV Navigation," IEEE ICCV Workshops 2022.
**arXiv:** https://arxiv.org/abs/2210.09727

WildNav is the closest published work to Module 2 of this project. It localizes a drone using only a pre-downloaded satellite tile map and a camera frame, with no GNSS at inference time. The pipeline matches local features (SIFT) between the drone frame and a satellite tile crop, applies RANSAC to filter outliers, and uses the homography to geolocate the camera footprint.

WildNav assumes a **nadir camera** (pointing straight down). This is a strong simplification: most consumer drones, including the DJI Mini 3 Pro used in this project, record with a forward-tilted gimbal (60° below horizon). Applying WildNav directly to a 60° frame would produce wrong matches because the perspective geometry of the drone image does not match the top-down satellite tile.

This project extends WildNav in three ways:

| Aspect | WildNav | This project |
| --- | --- | --- |
| Camera angle | Nadir (0°) | 60° oblique → IPM required |
| Tile source | Google Maps (zoom 17) | Esri World Imagery (zoom 18, finer GSD) |
| Feature matcher | SIFT + brute-force | SuperPoint + LightGlue (learned) |
| Tile coverage | Single tile | 3 × 3 mosaic (avoids boundary failures) |

### Inverse Perspective Mapping (IPM)

IPM is a geometric transformation that projects a perspective (oblique) image onto a top-down plane. It is well-established in autonomous driving for lane detection and bird's-eye-view perception (see Mallot et al., 1991; He et al., 2022 for modern formulations). In this project, IPM is adapted to the drone context: given drone altitude, camera tilt angle, and heading, each output pixel is computed by ray-casting from the camera through the ground plane and sampling the original frame. The output is a 512 × 512 px pseudo-nadir image at 0.5 m/px — the same GSD as an Esri zoom-18 tile — which enables direct feature matching against satellite imagery.

The IPM step is the key contribution of Module 2 relative to WildNav. Without it, the feature matching would fail on all 60°-tilted Mini 3 Pro frames because the visual geometry is incompatible with nadir satellite tiles.

---

## 10. Limitations and Future Directions

The main limitation of the retained pipeline is the gap between mean pooling (what we use) and VLAD with an aerial-domain vocabulary (what AnyLoc recommends). The paper's Table IV suggests this gap could be significant for aerial data: AnyLoc-VLAD-DINOv2 outperforms AnyLoc-GeM-DINOv2 by 0% vs 76.1% on Nardo-Air (GeM fails on this dataset while VLAD succeeds).

Other open directions include:

- **Camera heading as a prior**: the dead reckoning experiment in this project shows that optical flow can estimate drone speed accurately (~1.5% error over 115 frames) but accumulates large directional error without a heading source. Adding a magnetic heading reading would enable hybrid geometry + vision localization.
- **ViT-G14 with layer 31 value facet**: the current implementation uses ViT-S14 with mean pooling. Switching to the AnyLoc-recommended configuration would likely improve retrieval at the cost of higher compute.
- **VLAD with aerial vocabulary**: building a VLAD vocabulary from the three reference flights (v11, v12, v13) and using it for both reference encoding and query encoding is the most direct improvement aligned with the AnyLoc paper.
- **Satellite matching over vegetation**: Module 2 fails on 37/115 frames due to featureless vegetation areas where SuperPoint finds no discriminative keypoints. Template matching or learned dense matchers (e.g., LoFTR) might improve coverage in these regions.
- **Seasonal invariance**: the satellite tiles are a fixed snapshot; drone footage recorded in a different season (different foliage, snow) would degrade Module 2 matching. Methods that learn season-invariant features (e.g., trained on multi-season satellite pairs) could address this.
