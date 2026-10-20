# Reproduction scope and implementation notes

Source: the supplied `DGGI.pdf`, Section 4.1, DOI 10.1016/j.inffus.2024.102620. The journal volume is dated 2025; the paper was available online in August 2024.

## Paper versus supplied source

| Item | Paper | Supplied code |
| --- | --- | --- |
| Dataset scope | ImageNet and LDCT | Active defaults use `myimagenet`; complete LDCT/Fast-DDPM setup absent |
| Classification model | MoCo v2 ResNet-50 for ImageNet | ResNet-50 branch plus MoCo checkpoint loading |
| Reconstruction size | 224 × 224 | ImageFolder loader and diffusion crop use 224 × 224 |
| Main batch | 8 | Parser defaults to 8; sampler and index list are fixed at 8 |
| Optimizer and gradient distance | Adam, cosine | Default `zhu`: L-BFGS/L2; `ours`: Adam/cosine |
| Main learning rate | 0.1 | `ours`: 0.1; `zhu`: 1 |
| Iterations | 24,000 | `ours`: 24,000 per call; `zhu`: 300 per call |
| Initial reconstruction rounds | 4 | `--n_seed=4` |
| TV / BN / Group weights | 0.015 / 0.01 / 0.01 | 0.15 / 0.01 / 0.01 |
| Lambda | 1.8 | `alh=1.8` in the group-loss branch, subject to its activation condition |
| GPU | NVIDIA 3090 | CUDA-specific source; no CPU/Apple Silicon validation |

The original defaults, loss expressions, and experiment branches were preserved. The README command explicitly selects `ours` and overrides TV to 0.015; the default parser values remain unchanged.

## Group consistency condition

`get_best_pseudo` returns `x_group` and `x_max`. The final `GradientReconstructor` call in `reconstruct_image.py` passes only `x_group`.

In `inversefed/reconstruction_algorithms.py`, the group penalty is guarded by all of:

```python
self.config['Group'] > 0 and self.x_group != None and self.x_max != None
```

The constructor defaults `x_max` to `None`, so that condition is false for the supplied final call. This is a material reproduction issue. The snapshot does not establish whether the intended fix is to pass `x_max`, change the condition, or use a different loss formulation. No guessed algorithm change was made during formatting.

## Resource and execution assumptions

- None of the three default pretrained model files or datasets were supplied. Model training and weights are not reconstructed from the paper.
- ImageFolder labels must agree with the target model and the 1,000-class conditional diffusion model. Class folder names and subset selection must preserve that mapping.
- The ImageFolder loader uses a deterministic 80/20 split with seed 0. Assigning `.transform` to the returned Subsets does not replace the underlying dataset transform; augmentation behavior warrants separate validation.
- The ImageNet loader remains selected by the literal name `myimagenet`. Other dataset loaders do not establish compatibility with the ImageNet diffusion prior.
- Missing MoCo weights lead to the original message and continuation; the ResNet-50 branch initially constructs `pretrained=False`. Verify that the intended weights actually load before reporting results.
- Some boolean CLI options retain legacy parsing. Passing the string `False` does not reliably disable them.
- `--dryrun` does not skip diffusion sampling or resource loading.
- A batch size other than 8 needs coordinated changes in the sampler, labels, initialization, indexing, and reconstruction settings.

## Evaluation conventions

The supplied script reports the minimum LPIPS over batch members and computes SSIM on a tiled image. Compare these conventions with the intended evaluation protocol before reporting aggregate metrics. Several dataset-specific normalization paths and alternative precision branches also require validation.

## Other original issues retained

- `BNForwardFeatureHook` contains `register_forwad_hook`; that helper is not the hook used by the default BN-loss path.
- `inversefed/data/loss.py` contains unreachable references to `value` and `name` after `raise NotImplementedError()` in abstract methods.
- Optional model branches refer to missing `SRNet` and `BasicBlock` definitions.
- The optional `_get_meanstd` helper uses `trainset` instead of its `dataset` argument.

## Maintenance changes

- Replaced active machine-specific Windows paths with repository-relative defaults and environment overrides.
- Removed an unused top-level `import apex`; no code references to the Apex module remain. The existing method attribution to NVIDIA Apex is preserved.
- Changed the internal diffusion parser to `parse_args([])`. Previously it reparsed reconstruction options such as `--optim ours` and rejected them. The diffusion sampler now uses its existing model defaults and configured resource paths, without consuming the reconstruction CLI.
- Added a lightweight launcher and prerequisite checker. The launcher normalizes the working directory and creates resource/output directories.
- No experimental losses, iteration counts, model architecture defaults, class counts, or sampler defaults were changed.

## Validation boundary

Completed: Python 3.9 syntax parsing, compilation, Black formatting for maintained files, launcher help, path override tests, mocked launch arguments/working directory, parser separation, and source-only packaging checks.

Not completed: independent dependency installation, dataset/weight download, GPU reconstruction, end-to-end inference, or verification against the paper's numerical results. Source checks do not certify full reproducibility.
