"""Phase 5 explainability — the CAM itself, and the gate that judges it.

The gate statistic is tested against images whose answer is known by construction. A
border-artefact check that cannot be shown to DETECT a border artefact is decoration, and
it would pass silently forever.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.factory import ModelConfig, build_model
from src.xai.border_check import (OUTSIDE_MAX, RIM_MAX, image_ratios, mass_ratio,
                                  region_masks, summarise)
from src.xai.gradcam import (GradCAM, cam_correlation, pick_target_layer,
                             randomise_last_block, scalar_target)


# Every device this machine actually has. On the CPU-only dev box this is ["cpu"] and
# the cuda cases skip; on Kaggle it is both, and cell 1 of notebooks/phase5_gradcam.py
# runs this file BEFORE cell 2 does the real work — which is the whole point. A
# device-mismatch bug cannot be caught by a CPU-only fixture, so the fix is not a
# cleverer local test, it is running these tests in the context the artefact runs in.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# The pinned randomisation stream, generated on CPU with seed 7. These values must be
# identical on every device: if someone re-seeds a CUDA generator instead of copying
# CPU noise across, the numbers change on GPU and this test fails THERE, which is where
# the reproducibility of the gate would actually have been lost.
STREAM_SEED_7 = [-0.082013, 0.039563, 0.089891]


def ordinal_model():
    return build_model(ModelConfig(arch="efficientnet_b0", num_outputs=1,
                                   head="ordinal_regression", pretrained=False))


def softmax_model():
    return build_model(ModelConfig(arch="efficientnet_b0", num_outputs=5,
                                   head="softmax", pretrained=False))


def synthetic_fundus(size: int = 224, radius_frac: float = 0.45) -> np.ndarray:
    """A bright disc on black — the shape `retina_mask` is built to find."""
    img = np.zeros((size, size, 3), np.uint8)
    c, r = size // 2, int(size * radius_frac)
    cv_circle_centre = (c, c)
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - cv_circle_centre[1]) ** 2 + (xx - cv_circle_centre[0]) ** 2 <= r * r
    img[disc] = (120, 140, 160)
    return img


# --------------------------------------------------------------------- the scalar target

def test_ordinal_head_differentiates_the_scalar_not_an_argmax():
    """Arm E emits ONE output. `argmax` over a length-1 vector is always 0, so textbook
    Grad-CAM 'works' while meaning nothing."""
    out = torch.tensor([[1.7], [3.2]])
    assert torch.allclose(scalar_target(out), torch.tensor([1.7, 3.2]))


def test_softmax_head_is_explained_on_the_expected_grade():
    """DECISION-035 ranks on sum(p_i * i); explaining argmax would explain a different
    model than the one reported."""
    logits = torch.zeros(1, 5)
    logits[0, 4] = 100.0                      # essentially all mass on grade 4
    assert scalar_target(logits).item() == pytest.approx(4.0, abs=1e-3)
    assert scalar_target(torch.zeros(1, 5)).item() == pytest.approx(2.0)  # uniform -> 2


def test_a_one_dimensional_output_is_accepted():
    assert torch.allclose(scalar_target(torch.tensor([0.5, 1.5])),
                          torch.tensor([0.5, 1.5]))


# --------------------------------------------------------------------------- the CAM

def test_cam_has_input_resolution_and_is_normalised():
    m = ordinal_model()
    with GradCAM(m) as g:
        r = g(torch.randn(2, 3, 224, 224))
    assert r.cam.shape == (2, 224, 224)
    assert r.score.shape == (2,)
    assert r.cam.min() >= 0.0 and r.cam.max() <= 1.0


def test_gradcam_plus_plus_runs_on_the_ordinal_head():
    m = ordinal_model()
    with GradCAM(m, plus_plus=True) as g:
        r = g(torch.randn(1, 3, 224, 224))
    assert r.cam.shape == (1, 224, 224)


def test_cam_works_on_the_softmax_head_too():
    with GradCAM(softmax_model()) as g:
        r = g(torch.randn(1, 3, 224, 224))
    assert r.cam.shape == (1, 224, 224)


def test_cam_works_inside_no_grad():
    """Evaluation code is usually already inside torch.no_grad(); the backward pass must
    still run rather than failing on tensors that do not require grad."""
    m = ordinal_model()
    with torch.no_grad():
        with GradCAM(m) as g:
            r = g(torch.randn(1, 3, 224, 224))
    assert np.isfinite(r.cam).all()


def test_hooks_are_removed_on_exit():
    m = ordinal_model()
    layer = pick_target_layer(m)
    before = len(layer._forward_hooks)
    with GradCAM(m) as g:
        g(torch.randn(1, 3, 224, 224))
        assert len(layer._forward_hooks) == before + 1
    assert len(layer._forward_hooks) == before


def test_using_gradcam_without_the_context_manager_raises():
    with pytest.raises(RuntimeError, match="context manager"):
        GradCAM(ordinal_model())(torch.randn(1, 3, 224, 224))


def test_a_one_by_one_target_layer_is_refused():
    """A 1x1 map upsamples to a uniform heatmap, which looks like 'attends everywhere'
    rather than like a bug."""
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_head = nn.Conv2d(3, 4, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 1)

        def forward(self, x):
            return self.fc(self.pool(self.conv_head(x)).flatten(1))

    m = Tiny()
    with GradCAM(m, m.conv_head) as g:
        with pytest.raises(ValueError, match="spatial extent"):
            g(torch.randn(1, 3, 1, 1))


# ------------------------------------------------------- the gate statistic, by construction

def test_uniform_cam_scores_about_one_everywhere():
    """The ratio is normalised by area, so an indifferent map must score ~1.0 in every
    region — otherwise every threshold in DECISION-046 means something different."""
    bgr = synthetic_fundus()
    cam = np.ones((224, 224), float)
    ratios = image_ratios(cam, bgr)
    for region, v in ratios.items():
        assert v == pytest.approx(1.0, abs=1e-6), f"{region} = {v}"


def test_a_cam_concentrated_outside_the_disc_is_detected():
    """THE TEST THAT MAKES THE GATE MEAN SOMETHING. A map with all its mass in a corner —
    entirely outside the retina — must produce a large `outside` ratio and trip the gate."""
    bgr = synthetic_fundus()
    cam = np.zeros((224, 224), float)
    cam[:20, :20] = 1.0                       # a corner, outside any centred disc
    ratios = image_ratios(cam, bgr)
    assert ratios["outside"] > OUTSIDE_MAX, ratios
    assert ratios["outside"] > 1.5
    assert ratios["interior"] == pytest.approx(0.0)


def test_a_cam_concentrated_on_the_rim_is_detected():
    bgr = synthetic_fundus()
    regions = region_masks(bgr)
    cam = regions["rim"].astype(float)
    assert image_ratios(cam, bgr)["rim"] > RIM_MAX


def test_a_cam_concentrated_in_the_interior_passes():
    bgr = synthetic_fundus()
    regions = region_masks(bgr)
    cam = regions["interior"].astype(float)
    ratios = image_ratios(cam, bgr)
    assert ratios["rim"] == pytest.approx(0.0)
    assert ratios["outside"] == pytest.approx(0.0)


def test_regions_partition_the_image():
    regions = region_masks(synthetic_fundus())
    total = sum(r.sum() for r in regions.values())
    assert total == 224 * 224
    assert (regions["rim"] & regions["interior"]).sum() == 0
    assert (regions["rim"] & regions["outside"]).sum() == 0


def test_mass_ratio_of_an_empty_cam_is_zero_not_nan():
    regions = region_masks(synthetic_fundus())
    assert mass_ratio(np.zeros((224, 224)), regions["rim"]) == 0.0


# ------------------------------------------------------------------- the randomisation gate

def test_randomising_the_last_block_changes_the_cam():
    """A saliency map that survives randomising the target layer is measuring the image,
    not the model. The check stands on its own logic; the citation usually given for it
    is unverified and deliberately not repeated."""
    m = ordinal_model()
    x = torch.randn(2, 3, 224, 224)
    with GradCAM(m) as g:
        a = g(x).cam
    with GradCAM(randomise_last_block(m, seed=1)) as g:
        b = g(x).cam
    assert abs(cam_correlation(a[0], b[0])) < 0.9


def test_randomisation_deep_copies_and_leaves_the_original_intact():
    m = ordinal_model()
    before = pick_target_layer(m).weight.detach().clone()
    randomise_last_block(m, seed=0)
    assert torch.equal(pick_target_layer(m).weight, before)


def test_identical_cams_correlate_at_one():
    a = np.random.default_rng(0).random((16, 16))
    assert cam_correlation(a, a) == pytest.approx(1.0)


def test_two_constant_cams_correlate_at_one_rather_than_nan():
    """NaN would be dropped from a median and turn a failing gate into a passing one."""
    assert cam_correlation(np.ones((8, 8)), np.ones((8, 8))) == 1.0


# ------------------------------------------------------------------------- the verdict

def rows(rim, outside, interior=1.0, labels=(0, 1, 2, 3, 4)):
    return [{"rim": rim, "outside": outside, "interior": interior, "label": g}
            for g in labels]


def test_summary_passes_when_every_statistic_is_inside_its_threshold():
    s = summarise(rows(1.0, 0.1), randomisation=[0.05, 0.1])
    assert s["passed"] and all(s["gates"].values())


def test_a_single_failing_statistic_fails_the_whole_gate():
    assert not summarise(rows(2.0, 0.1))["passed"]
    assert not summarise(rows(1.0, 0.9))["passed"]
    assert not summarise(rows(1.0, 0.1), randomisation=[0.95])["passed"]


def test_a_sample_with_no_severe_images_fails_rather_than_passing_by_omission():
    """The grade 3-4 gate cannot be cleared by a sample containing no grade 3-4 images."""
    s = summarise(rows(1.0, 0.1, labels=(0, 0, 1, 2)))
    assert s["n_severe"] == 0
    assert not s["gates"]["rim_severe"]
    assert not s["passed"]


def test_severe_rim_bias_is_caught_even_when_the_overall_median_is_clean():
    """73.5% of images are grade 0, so a rim bias confined to severe grades would be
    invisible in an overall median — which is exactly why it has its own gate."""
    clean = [{"rim": 0.9, "outside": 0.1, "interior": 1.0, "label": 0} for _ in range(40)]
    biased = [{"rim": 3.0, "outside": 0.1, "interior": 1.0, "label": 4} for _ in range(5)]
    s = summarise(clean + biased)
    assert s["gates"]["rim"], "the overall median should look clean here"
    assert not s["gates"]["rim_severe"]
    assert not s["passed"]


# ---------------------------------------------------------------- device parity (regression)

@pytest.mark.parametrize("device", DEVICES)
def test_randomise_last_block_works_on_every_available_device(device):
    """REGRESSION: Phase 5 cell 2 died with

        RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

    after all 24 tests in this file passed on CPU. A torch.Generator is device-bound, so
    a CPU generator cannot seed normal_() on a CUDA parameter.
    """
    m = ordinal_model().to(device)
    clone = randomise_last_block(m, seed=7)
    w = pick_target_layer(clone).weight
    assert w.device.type == device
    assert torch.isfinite(w).all()


@pytest.mark.parametrize("device", DEVICES)
def test_randomisation_stream_is_identical_on_every_device(device):
    """The seed is the only reason the gate repeats, so the stream is pinned to CPU and
    copied across rather than re-seeded per device — the two produce DIFFERENT streams
    for the same seed, which would make the gate irreproducible across machines."""
    clone = randomise_last_block(ordinal_model().to(device), seed=7)
    got = [round(float(v), 6) for v in
           pick_target_layer(clone).weight.detach().flatten()[:3].cpu()]
    assert got == STREAM_SEED_7, f"stream changed on {device}: {got}"


@pytest.mark.parametrize("device", DEVICES)
def test_noise_is_generated_on_cpu_whatever_device_the_model_is_on(monkeypatch, device):
    """The invariant that broke, asserted directly. Vacuous on a CPU-only box and
    meaningful on GPU — which is the honest shape of this bug."""
    seen = []
    real = torch.Tensor.normal_

    def spy(self, *a, **kw):
        seen.append(self.device.type)
        return real(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "normal_", spy)
    randomise_last_block(ordinal_model().to(device), seed=0)
    assert seen, "no noise was generated at all"
    assert set(seen) == {"cpu"}, f"noise generated on {set(seen)}"


@pytest.mark.parametrize("device", DEVICES)
def test_cam_runs_end_to_end_on_every_available_device(device):
    m = ordinal_model().to(device)
    with GradCAM(m) as g:
        r = g(torch.randn(2, 3, 224, 224, device=device))
    assert r.cam.shape == (2, 224, 224)
    assert np.isfinite(r.cam).all()


@pytest.mark.parametrize("device", DEVICES)
def test_the_full_randomisation_gate_path_on_every_device(device):
    """Exactly what cell 2 does: real CAM, randomised CAM, correlation between them."""
    m = ordinal_model().to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    with GradCAM(m) as g:
        real = g(x).cam
    with GradCAM(randomise_last_block(m, seed=20260825).to(device)) as g:
        rand = g(x).cam
    c = cam_correlation(real[0], rand[0])
    assert np.isfinite(c)


def test_a_zero_dimensional_parameter_is_zeroed_not_crashed():
    """Some layers carry 0-dim parameters; normal_ on them is meaningless."""
    import torch.nn as nn

    class WithScalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_head = nn.Conv2d(3, 4, 3, padding=1)
            self.conv_head.gain = nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return self.conv_head(x).mean(dim=(1, 2, 3), keepdim=False)[:, None]

    m = WithScalar()
    clone = randomise_last_block(m, seed=0)
    assert float(pick_target_layer(clone).gain.detach()) == 0.0


def test_randomisation_survives_a_half_precision_parameter():
    """`normal_` has no CPU kernel for float16, so a half checkpoint would fail here for
    a second, unrelated device-ish reason."""
    m = ordinal_model()
    layer = pick_target_layer(m)
    layer.half()
    clone = randomise_last_block(m, seed=0)
    w = pick_target_layer(clone).weight
    assert w.dtype == torch.float16 and torch.isfinite(w).all()
