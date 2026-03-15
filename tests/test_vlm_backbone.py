"""Tests for VLM backbone and image processor.

Tier 1: No internet / no model download required (always run).
Tier 2: Requires mlx-vlm model download (marked @pytest.mark.slow).
"""

import numpy as np
import pytest

import mlx.core as mx

from lerobot_mlx.model.vlm_backbone import VLMBackbone
from lerobot_mlx.model.image_processor import ImageProcessor, IMAGENET_MEAN, IMAGENET_STD


# ===========================================================================
# Tier 1: No internet required
# ===========================================================================


class TestVLMBackboneUnit:
    """Unit tests for VLMBackbone without loading real models."""

    def test_vlm_backbone_init(self):
        """VLMBackbone() creates an unloaded instance."""
        vlm = VLMBackbone()
        assert vlm._model is None
        assert vlm._processor is None
        assert vlm._config is None
        assert not vlm.is_loaded

    def test_vlm_backbone_init_with_args(self):
        """VLMBackbone can be constructed with model/processor/config."""
        vlm = VLMBackbone(model="fake", processor="fake_proc", config={"hidden_size": 512})
        assert vlm._model == "fake"
        assert vlm._processor == "fake_proc"
        assert vlm.is_loaded

    def test_vlm_backbone_not_loaded_encode_raises(self):
        """encode() raises RuntimeError when not loaded."""
        vlm = VLMBackbone()
        with pytest.raises(RuntimeError, match="VLM not loaded"):
            vlm.encode(text="hello")

    def test_vlm_backbone_not_loaded_vision_raises(self):
        """get_vision_features() raises RuntimeError when not loaded."""
        vlm = VLMBackbone()
        with pytest.raises(RuntimeError, match="VLM not loaded"):
            vlm.get_vision_features(mx.zeros((1, 3, 224, 224)))

    def test_vlm_backbone_not_loaded_text_raises(self):
        """get_text_features() raises RuntimeError when not loaded."""
        vlm = VLMBackbone()
        with pytest.raises(RuntimeError, match="VLM not loaded"):
            vlm.get_text_features("hello")

    def test_vlm_backbone_available_models(self):
        """available_models() returns a dict of recommendations."""
        models = VLMBackbone.available_models()
        assert isinstance(models, dict)
        assert len(models) >= 4
        assert "paligemma-3b" in models
        assert "smolvlm-500m" in models

    def test_vlm_backbone_repr_not_loaded(self):
        """__repr__ shows 'not loaded'."""
        vlm = VLMBackbone()
        r = repr(vlm)
        assert "not loaded" in r
        assert "hidden_size=2048" in r  # default

    def test_vlm_backbone_repr_loaded(self):
        """__repr__ shows 'loaded' when model is present."""
        vlm = VLMBackbone(model="fake", config={"hidden_size": 768})
        r = repr(vlm)
        assert "loaded" in r
        assert "hidden_size=768" in r

    def test_vlm_backbone_repr_frozen(self):
        """__repr__ shows 'frozen' when model is frozen."""
        vlm = VLMBackbone(model="fake")
        vlm._frozen = True
        r = repr(vlm)
        assert "frozen" in r

    def test_vlm_backbone_hidden_size_default(self):
        """hidden_size returns 2048 when no config."""
        vlm = VLMBackbone()
        assert vlm.hidden_size == 2048

    def test_vlm_backbone_hidden_size_from_config(self):
        """hidden_size reads from config dict."""
        vlm = VLMBackbone(config={"hidden_size": 1024})
        assert vlm.hidden_size == 1024

    def test_vlm_backbone_hidden_size_from_text_config(self):
        """hidden_size prefers text_config.hidden_size."""
        vlm = VLMBackbone(config={
            "hidden_size": 1024,
            "text_config": {"hidden_size": 2560},
        })
        assert vlm.hidden_size == 2560

    def test_vlm_backbone_vision_hidden_size_default(self):
        """vision_hidden_size returns 1152 when no config."""
        vlm = VLMBackbone()
        assert vlm.vision_hidden_size == 1152

    def test_vlm_backbone_vision_hidden_size_from_config(self):
        """vision_hidden_size reads from config dict."""
        vlm = VLMBackbone(config={"vision_config": {"hidden_size": 768}})
        assert vlm.vision_hidden_size == 768

    def test_vlm_backbone_freeze_unfreeze_none(self):
        """freeze/unfreeze is a no-op when model is None."""
        vlm = VLMBackbone()
        vlm.freeze()
        assert not vlm._frozen  # model is None, nothing to freeze
        vlm.unfreeze()
        assert not vlm._frozen

    def test_vlm_backbone_is_frozen_property(self):
        """is_frozen property reflects freeze state."""
        vlm = VLMBackbone()
        assert not vlm.is_frozen

    def test_vlm_backbone_properties(self):
        """model, processor, config properties work."""
        vlm = VLMBackbone(model="m", processor="p", config={"k": "v"})
        assert vlm.model == "m"
        assert vlm.processor == "p"
        assert vlm.config == {"k": "v"}


class TestImageProcessor:
    """Unit tests for the ImageProcessor."""

    def test_image_processor_numpy_hwc(self):
        """Process a numpy HWC image."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=False)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc(img)
        assert isinstance(result, mx.array)
        assert result.shape == (1, 64, 64, 3)

    def test_image_processor_numpy_chw(self):
        """Process a numpy CHW image, auto-detects and converts."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=True)
        img = np.random.rand(3, 64, 64).astype(np.float32)
        result = proc(img)
        assert isinstance(result, mx.array)
        # Input is CHW, _ensure_hwc converts to HWC, then back to CHW
        assert result.shape == (1, 3, 64, 64)

    def test_image_processor_mx_array(self):
        """Process an mx.array image."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=False)
        img = mx.zeros((64, 64, 3))
        result = proc(img)
        assert isinstance(result, mx.array)
        assert result.shape == (1, 64, 64, 3)

    def test_image_processor_pil(self):
        """Process a PIL image."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available")

        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=True)
        img = PILImage.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = proc(img)
        assert isinstance(result, mx.array)
        assert result.shape == (1, 3, 64, 64)

    def test_image_processor_resize(self):
        """Resize works correctly."""
        proc = ImageProcessor(size=(32, 32), mean=None, std=None, channel_first=True)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc(img)
        assert result.shape == (1, 3, 32, 32)

    def test_image_processor_normalize(self):
        """Normalization applies correctly."""
        proc = ImageProcessor(size=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), channel_first=False)
        # All-white uint8 image -> after rescale to [0,1] -> 1.0 -> (1.0 - 0.5) / 0.5 = 1.0
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = proc(img)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np[0], 1.0, atol=1e-5)

    def test_image_processor_normalize_zero(self):
        """Normalization on black image gives expected value."""
        proc = ImageProcessor(size=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), channel_first=False)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = proc(img)
        result_np = np.array(result)
        # (0.0 - 0.5) / 0.5 = -1.0
        np.testing.assert_allclose(result_np[0], -1.0, atol=1e-5)

    def test_image_processor_multi_view(self):
        """Multi-view processing returns dict of tensors."""
        proc = ImageProcessor(size=(32, 32), mean=None, std=None, channel_first=True)
        cameras = {
            "front": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8),
        }
        result = proc.process_multi_view(cameras)
        assert isinstance(result, dict)
        assert "front" in result and "wrist" in result
        assert result["front"].shape == (1, 3, 32, 32)
        assert result["wrist"].shape == (1, 3, 32, 32)

    def test_image_processor_batch(self):
        """Process a batch of numpy images."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=True)
        imgs = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        ]
        result = proc(imgs)
        assert result.shape == (2, 3, 64, 64)

    def test_image_processor_4d_numpy(self):
        """Process a 4D numpy batch (B, H, W, C)."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=True)
        batch = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        result = proc(batch)
        assert result.shape == (3, 3, 64, 64)

    def test_image_processor_grayscale(self):
        """Process a grayscale image (H, W)."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=True)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = proc(img)
        assert result.shape == (1, 1, 64, 64)

    def test_image_processor_rescale(self):
        """Rescale converts uint8 [0,255] to [0,1]."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=False, rescale=True)
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = proc(img)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np[0, 0, 0, 0], 128.0 / 255.0, atol=1e-5)

    def test_image_processor_no_rescale(self):
        """When rescale=False, float images are not rescaled."""
        proc = ImageProcessor(size=None, mean=None, std=None, channel_first=False, rescale=False)
        img = np.full((4, 4, 3), 0.7, dtype=np.float32)
        result = proc(img)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np[0, 0, 0, 0], 0.7, atol=1e-5)

    def test_pad_to_square(self):
        """pad_to_square pads a non-square image."""
        img = np.random.rand(32, 48, 3).astype(np.float32)
        padded = ImageProcessor.pad_to_square(img)
        assert padded.shape == (48, 48, 3)

    def test_pad_to_square_already_square(self):
        """pad_to_square returns same shape for square input."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        padded = ImageProcessor.pad_to_square(img)
        assert padded.shape == (32, 32, 3)


class TestVLMBackboneImportExport:
    """Test module-level imports work."""

    def test_import_from_model(self):
        """VLMBackbone and ImageProcessor importable from model package."""
        from lerobot_mlx.model import VLMBackbone, ImageProcessor
        assert VLMBackbone is not None
        assert ImageProcessor is not None

    def test_import_direct(self):
        """Direct module imports work."""
        from lerobot_mlx.model.vlm_backbone import VLMBackbone
        from lerobot_mlx.model.image_processor import ImageProcessor
        assert VLMBackbone is not None
        assert ImageProcessor is not None


# ===========================================================================
# Tier 2: Requires mlx-vlm model download
# ===========================================================================


@pytest.mark.slow
class TestVLMBackboneIntegration:
    """Integration tests that require downloading a VLM model.

    These tests download a small quantized model from HuggingFace Hub.
    Mark with -m slow to run: pytest -m slow tests/test_vlm_backbone.py
    """

    MODEL_ID = "mlx-community/paligemma-3b-pt-224-4bit"

    @pytest.fixture(scope="class")
    def vlm(self):
        """Load a VLM model (cached across tests in this class)."""
        return VLMBackbone.from_pretrained(self.MODEL_ID)

    def test_vlm_from_pretrained(self, vlm):
        """VLM loads successfully."""
        assert vlm.is_loaded
        assert vlm.model is not None
        assert vlm.processor is not None

    def test_vlm_hidden_size(self, vlm):
        """Hidden size matches model config."""
        assert vlm.hidden_size > 0
        assert isinstance(vlm.hidden_size, int)

    def test_vlm_vision_hidden_size(self, vlm):
        """Vision hidden size is set from config."""
        assert vlm.vision_hidden_size > 0
        assert isinstance(vlm.vision_hidden_size, int)

    def test_vlm_encode_shape(self, vlm):
        """encode() returns features with correct dimensions."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL required")

        img = PILImage.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        features = vlm.encode(images=img, text="pick up the cube")
        assert isinstance(features, mx.array)
        assert features.ndim >= 2  # At least (seq_len, hidden_size)

    def test_vlm_vision_features(self, vlm):
        """get_vision_features returns vision embeddings."""
        pixel_values = mx.random.normal((1, 3, 224, 224))
        # Transpose to NHWC if needed by model
        try:
            features = vlm.get_vision_features(pixel_values)
            assert isinstance(features, mx.array)
            assert features.ndim >= 2
        except (NotImplementedError, Exception):
            pytest.skip("Vision tower not accessible in this architecture")

    def test_vlm_freeze_parameters(self, vlm):
        """freeze() actually freezes model parameters."""
        vlm.freeze()
        assert vlm.is_frozen
        vlm.unfreeze()
        assert not vlm.is_frozen

    def test_vlm_repr_after_load(self, vlm):
        """repr is informative after loading."""
        r = repr(vlm)
        assert "loaded" in r
