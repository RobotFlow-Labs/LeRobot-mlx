"""Universal VLM backbone wrapper using mlx-vlm.

Provides a clean interface to load any VLM supported by mlx-vlm and extract
features for VLA policy conditioning. Supports PaliGemma, SmolVLM, Qwen2-VL,
LLaVA, and 40+ other architectures.

Usage:
    from lerobot_mlx.model.vlm_backbone import VLMBackbone

    vlm = VLMBackbone.from_pretrained("mlx-community/paligemma-3b-pt-224-4bit")
    features = vlm.encode(images=images, text="pick up the cube")
"""

import logging
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class VLMBackbone:
    """Universal Vision-Language Model backbone for VLA policies.

    Wraps any mlx-vlm model to provide:
    - Image feature extraction
    - Text feature extraction
    - Combined vision+language embeddings
    - Freezing/unfreezing for training
    """

    def __init__(self, model=None, processor=None, config=None):
        """Initialize VLMBackbone.

        Args:
            model: An mlx-vlm model instance (or None for unloaded state).
            processor: The associated processor/tokenizer (or None).
            config: Model configuration dict (or None).
        """
        self._model = model
        self._processor = processor
        self._config = config
        self._frozen = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        quantize: Optional[str] = None,
        lazy: bool = False,
        **kwargs,
    ) -> "VLMBackbone":
        """Load a pretrained VLM from HuggingFace Hub via mlx-vlm.

        Args:
            model_id: HF model ID (e.g., "mlx-community/paligemma-3b-pt-224-4bit")
            quantize: Optional quantization ("4bit", "8bit") -- currently unused,
                      pass quantized model IDs instead.
            lazy: If True, defer weight loading until first use.
            **kwargs: Additional keyword arguments passed to mlx_vlm.load().

        Returns:
            VLMBackbone instance ready for feature extraction.
        """
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError:
            raise ImportError(
                "mlx-vlm is required for VLM backbone loading. "
                "Install with: pip install mlx-vlm"
            )

        logger.info(f"Loading VLM: {model_id}")
        model, processor = load(model_id, lazy=lazy, **kwargs)
        config = load_config(model_id)

        backbone = cls(model=model, processor=processor, config=config)
        logger.info(f"VLM loaded: {model_id} (hidden_size={backbone.hidden_size})")
        return backbone

    @property
    def hidden_size(self) -> int:
        """Get the VLM's hidden dimension size.

        Searches config dict for text_config.hidden_size, then hidden_size,
        with a fallback of 2048.
        """
        if self._config and isinstance(self._config, dict):
            text_config = self._config.get("text_config", {})
            if isinstance(text_config, dict) and "hidden_size" in text_config:
                return text_config["hidden_size"]
            if "hidden_size" in self._config:
                return self._config["hidden_size"]
        return 2048  # Default fallback

    @property
    def vision_hidden_size(self) -> int:
        """Get the vision encoder's hidden dimension.

        Searches config dict for vision_config.hidden_size, with a fallback of 1152.
        """
        if self._config and isinstance(self._config, dict):
            vision_config = self._config.get("vision_config", {})
            if isinstance(vision_config, dict) and "hidden_size" in vision_config:
                return vision_config["hidden_size"]
        return 1152  # Default fallback

    @property
    def is_loaded(self) -> bool:
        """Whether a model has been loaded."""
        return self._model is not None

    @property
    def model(self):
        """Access the underlying mlx-vlm model."""
        return self._model

    @property
    def processor(self):
        """Access the underlying processor/tokenizer."""
        return self._processor

    @property
    def config(self) -> dict:
        """Access the model configuration dict."""
        return self._config

    def encode(self, images=None, text=None) -> mx.array:
        """Encode images and/or text into feature embeddings.

        Args:
            images: PIL Image, numpy array, or mx.array.
                    Supports (B, C, H, W), (B, H, W, C), or a single image.
            text: str or list of str

        Returns:
            mx.array of shape (B, seq_len, hidden_size)

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("VLM not loaded. Call from_pretrained() first.")

        inputs = self._prepare_inputs(images, text)
        embeddings = self._extract_features(inputs)
        return embeddings

    def get_vision_features(self, images) -> mx.array:
        """Extract vision-only features from images.

        Args:
            images: (B, C, H, W) or (B, H, W, C) image tensor, PIL images,
                    or numpy array.

        Returns:
            mx.array of shape (B, num_patches, vision_hidden_size)

        Raises:
            RuntimeError: If no model is loaded.
            NotImplementedError: If the model has no accessible vision_tower.
        """
        if not self.is_loaded:
            raise RuntimeError("VLM not loaded.")

        pixel_values = self._process_images(images)

        if hasattr(self._model, "vision_tower"):
            features = self._model.vision_tower(pixel_values)
            if isinstance(features, tuple):
                features = features[0]
            return features

        if hasattr(self._model, "vision_model"):
            features = self._model.vision_model(pixel_values)
            if isinstance(features, tuple):
                features = features[0]
            return features

        raise NotImplementedError(
            "This VLM architecture doesn't expose vision_tower or vision_model"
        )

    def get_text_features(self, text) -> mx.array:
        """Extract text-only features (token embeddings).

        Args:
            text: str or list of str

        Returns:
            mx.array of shape (B, seq_len, hidden_size)

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("VLM not loaded.")

        if isinstance(text, str):
            text = [text]

        input_ids = self._tokenize(text)

        # Get text embeddings from the language model's embed_tokens
        if hasattr(self._model, "language_model"):
            lm = self._model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
                return lm.model.embed_tokens(input_ids)
            if hasattr(lm, "embed_tokens"):
                return lm.embed_tokens(input_ids)

        raise NotImplementedError(
            "Could not find embed_tokens on the language model."
        )

    def freeze(self):
        """Freeze all VLM parameters (for fine-tuning only action heads)."""
        if self._model is not None:
            self._model.freeze()
            self._frozen = True

    def unfreeze(self):
        """Unfreeze VLM parameters for full fine-tuning."""
        if self._model is not None:
            self._model.unfreeze()
            self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Whether the model parameters are frozen."""
        return self._frozen

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]) -> mx.array:
        """Tokenize a list of strings into input_ids."""
        if self._processor is not None:
            if hasattr(self._processor, "tokenizer"):
                tokens = self._processor.tokenizer(
                    texts, return_tensors="np", padding=True
                )
                return mx.array(tokens["input_ids"])
            # Some processors are tokenizers themselves
            if callable(self._processor):
                tokens = self._processor(texts, return_tensors="np", padding=True)
                if isinstance(tokens, dict) and "input_ids" in tokens:
                    return mx.array(tokens["input_ids"])
        raise RuntimeError("No tokenizer available on processor.")

    def _prepare_inputs(self, images, text) -> dict:
        """Prepare inputs using the model's processor."""
        processed = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            processed["input_ids"] = self._tokenize(text)

        if images is not None:
            processed["pixel_values"] = self._process_images(images)

        return processed

    def _process_images(self, images) -> mx.array:
        """Process images to pixel_values tensor."""
        if isinstance(images, mx.array):
            return images
        if isinstance(images, np.ndarray):
            return mx.array(images)

        # Try PIL
        try:
            from PIL import Image as PILImage

            if isinstance(images, PILImage.Image):
                images = [images]
            if isinstance(images, (list, tuple)) and len(images) > 0:
                if isinstance(images[0], PILImage.Image):
                    if (
                        self._processor is not None
                        and hasattr(self._processor, "image_processor")
                    ):
                        processed = self._processor.image_processor(
                            images, return_tensors="np"
                        )
                        if isinstance(processed, dict):
                            return mx.array(processed["pixel_values"])
                        return mx.array(np.array(processed))
                    # Fallback: convert PIL to numpy
                    arrays = [np.array(img, dtype=np.float32) for img in images]
                    return mx.array(np.stack(arrays))
        except ImportError:
            pass

        return mx.array(np.asarray(images, dtype=np.float32))

    def _extract_features(self, inputs: dict) -> mx.array:
        """Extract features from the VLM (not generate text).

        Uses the model's forward pass to get hidden states. Tries multiple
        common patterns used by different VLM architectures.
        """
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")

        # Pattern 1: get_input_embeddings (PaliGemma, many VLMs)
        if hasattr(self._model, "get_input_embeddings"):
            try:
                result = self._model.get_input_embeddings(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )
                # result may be a namedtuple or object with inputs_embeds
                if hasattr(result, "inputs_embeds"):
                    return result.inputs_embeds
                if isinstance(result, mx.array):
                    return result
                if isinstance(result, tuple):
                    return result[0]
            except Exception as e:
                logger.debug(f"get_input_embeddings failed: {e}")

        # Pattern 2: Direct __call__ with output_hidden_states
        if hasattr(self._model, "__call__"):
            try:
                kwargs = {}
                if input_ids is not None:
                    kwargs["input_ids"] = input_ids
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values

                output = self._model(**kwargs)

                if hasattr(output, "hidden_states") and output.hidden_states:
                    return output.hidden_states[-1]
                if isinstance(output, tuple) and len(output) > 0:
                    return output[0]
                if isinstance(output, mx.array):
                    return output
            except Exception as e:
                logger.debug(f"Direct model call failed: {e}")

        # Pattern 3: vision features fallback
        if pixel_values is not None:
            for attr in ("vision_tower", "vision_model"):
                if hasattr(self._model, attr):
                    features = getattr(self._model, attr)(pixel_values)
                    if isinstance(features, tuple):
                        features = features[0]
                    return features

        raise RuntimeError(
            "Could not extract features from VLM. "
            "The model architecture may not be compatible."
        )

    @staticmethod
    def available_models() -> dict[str, str]:
        """List recommended VLM models for robotics use cases.

        Returns:
            Dict mapping short names to HuggingFace model IDs.
        """
        return {
            "paligemma-3b": "google/paligemma-3b-pt-224",
            "paligemma-3b-4bit": "mlx-community/paligemma-3b-pt-224-4bit",
            "smolvlm-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "qwen2-vl-2b": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "llava-1.5-7b": "mlx-community/llava-1.5-7b-4bit",
        }

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        frozen = ", frozen" if self._frozen else ""
        return f"VLMBackbone({status}, hidden_size={self.hidden_size}{frozen})"
