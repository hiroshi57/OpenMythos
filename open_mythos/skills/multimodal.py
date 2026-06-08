"""
Sprint 48 — マルチモーダル統合

Hermes Skills: clip / llava / stable-diffusion / segment-anything
ref: skills/multimodal/*-SKILL.md

画像・音声・テキストのマルチモーダル処理を OpenMythos に統合する。
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CLIP 画像-テキスト埋め込み
# ---------------------------------------------------------------------------

@dataclass
class CLIPEmbedding:
    """CLIP 埋め込みベクトル。"""
    vector: List[float]
    modality: str          # "image" | "text"
    dim: int

    def similarity(self, other: "CLIPEmbedding") -> float:
        """コサイン類似度を計算する。"""
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        na = math.sqrt(sum(x * x for x in self.vector)) or 1e-9
        nb = math.sqrt(sum(x * x for x in other.vector)) or 1e-9
        return dot / (na * nb)


class CLIPModel:
    """CLIP モデルラッパー。

    `transformers` + `PIL` がある場合は本物を使用し、
    ない場合はダミー埋め込みを返す。
    """

    MODELS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-large-patch14": 768,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._dim = self.MODELS.get(model_name, 512)
        self._model = None
        self._processor = None
        self._native = False
        try:
            from transformers import CLIPModel as _CM, CLIPProcessor  # type: ignore
            self._model = _CM.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._native = True
        except (ImportError, Exception):
            pass

    def encode_text(self, texts: List[str]) -> List[CLIPEmbedding]:
        """テキストを CLIP ベクトルに変換する。"""
        if self._native:
            try:
                import torch
                inputs = self._processor(text=texts, return_tensors="pt", padding=True)
                with torch.no_grad():
                    feats = self._model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return [
                    CLIPEmbedding(vector=feats[i].tolist(), modality="text", dim=self._dim)
                    for i in range(len(texts))
                ]
            except Exception:
                pass
        # fallback: ダミー正規化ベクトル
        return [
            CLIPEmbedding(
                vector=self._dummy_vector(text, self._dim),
                modality="text",
                dim=self._dim,
            )
            for text in texts
        ]

    def encode_image_b64(self, image_b64: str) -> CLIPEmbedding:
        """Base64 画像を CLIP ベクトルに変換する。"""
        if self._native:
            try:
                import io, torch
                from PIL import Image  # type: ignore
                img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                inputs = self._processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return CLIPEmbedding(vector=feats[0].tolist(), modality="image", dim=self._dim)
            except Exception:
                pass
        v = self._dummy_vector(image_b64[:32], self._dim)
        return CLIPEmbedding(vector=v, modality="image", dim=self._dim)

    def zero_shot_classify(
        self, image_b64: str, labels: List[str]
    ) -> Dict[str, float]:
        """ゼロショット画像分類を行う。"""
        img_emb = self.encode_image_b64(image_b64)
        text_embs = self.encode_text(labels)
        sims = {lbl: img_emb.similarity(emb) for lbl, emb in zip(labels, text_embs)}
        total = sum(math.exp(s * 10) for s in sims.values()) or 1.0
        return {k: round(math.exp(v * 10) / total, 4) for k, v in sims.items()}

    @staticmethod
    def _dummy_vector(seed: str, dim: int) -> List[float]:
        import hashlib
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        v = [(((h >> i) & 0xFF) / 255.0 - 0.5) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# LLaVA 視覚言語モデル
# ---------------------------------------------------------------------------

@dataclass
class VisionChatMessage:
    """画像 + テキストメッセージ。"""
    role: str               # "user" | "assistant"
    text: str
    image_b64: Optional[str] = None


@dataclass
class VisionChatResult:
    """視覚言語モデルの応答。"""
    response: str
    model: str
    tokens_used: int


class LLaVAModel:
    """LLaVA 視覚言語モデルラッパー。

    `transformers` がある場合は LLaVA モデルを使用し、
    ない場合はダミー応答を返す。
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._native = False
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor  # type: ignore
            self._model = LlavaForConditionalGeneration.from_pretrained(model_name)
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._native = True
        except (ImportError, Exception):
            pass

    def chat(
        self,
        messages: List[VisionChatMessage],
        max_new_tokens: int = 256,
    ) -> VisionChatResult:
        """画像を含む会話を処理する。"""
        if self._native and self._model:
            try:
                import torch, io
                from PIL import Image as PILImage  # type: ignore
                text_parts = []
                images = []
                for msg in messages:
                    if msg.image_b64:
                        img = PILImage.open(io.BytesIO(base64.b64decode(msg.image_b64))).convert("RGB")
                        images.append(img)
                        text_parts.append(f"<image>\n{msg.text}")
                    else:
                        text_parts.append(msg.text)
                prompt = "\n".join(text_parts)
                inputs = self._processor(text=prompt, images=images or None, return_tensors="pt")
                with torch.no_grad():
                    out = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
                response = self._processor.decode(out[0], skip_special_tokens=True)
                return VisionChatResult(response=response, model=self.model_name, tokens_used=len(out[0]))
            except Exception:
                pass
        # fallback
        last_msg = messages[-1] if messages else VisionChatMessage(role="user", text="")
        has_image = any(m.image_b64 for m in messages)
        response = f"[LLaVA mock] I {'see an image and' if has_image else ''}answer: {last_msg.text[:50]}"
        return VisionChatResult(response=response, model=self.model_name, tokens_used=len(response.split()))

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# Stable Diffusion 画像生成
# ---------------------------------------------------------------------------

@dataclass
class DiffusionRequest:
    """画像生成リクエスト。"""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1
    model: str = "stabilityai/stable-diffusion-2-1"


@dataclass
class DiffusionResult:
    """画像生成結果。"""
    image_b64: str          # PNG base64
    prompt: str
    seed: int
    steps: int
    model: str
    width: int
    height: int


class StableDiffusionGenerator:
    """Stable Diffusion テキスト→画像生成。

    `diffusers` がある場合は本物のパイプラインを使用し、
    ない場合はダミー PNG を返す。
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._pipe = None
        self._native = False
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore
            self._SD = StableDiffusionPipeline
            self._native = True
        except ImportError:
            self._SD = None

    def generate(self, req: DiffusionRequest) -> DiffusionResult:
        """テキストプロンプトから画像を生成する。"""
        import random
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32)
        if self._native and self._SD:
            try:
                import torch
                pipe = self._SD.from_pretrained(req.model, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32)
                pipe = pipe.to(self.device)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                image = pipe(
                    req.prompt,
                    negative_prompt=req.negative_prompt or None,
                    width=req.width, height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                ).images[0]
                import io
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                return DiffusionResult(
                    image_b64=img_b64, prompt=req.prompt, seed=seed,
                    steps=req.steps, model=req.model, width=req.width, height=req.height,
                )
            except Exception:
                pass
        # fallback: 1x1 白 PNG
        PNG_1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        return DiffusionResult(
            image_b64=PNG_1x1, prompt=req.prompt, seed=seed,
            steps=req.steps, model=req.model, width=req.width, height=req.height,
        )

    @property
    def is_native(self) -> bool:
        return self._native


# ---------------------------------------------------------------------------
# Segment Anything (SAM)
# ---------------------------------------------------------------------------

@dataclass
class SegmentRequest:
    """セグメンテーションリクエスト。"""
    image_b64: str
    points: List[Tuple[int, int]] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    multimask: bool = True


@dataclass
class SegmentMask:
    """セグメンテーションマスク。"""
    mask_b64: str           # PNG base64 (binary mask)
    score: float
    area: int


@dataclass
class SegmentResult:
    """セグメンテーション結果。"""
    masks: List[SegmentMask]
    n_masks: int


class SAMSegmenter:
    """Segment Anything Model (SAM) ラッパー。

    `segment-anything` ライブラリがある場合は本物を使用し、
    ない場合はダミー結果を返す。
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str = "",
        device: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self._native = False
        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
            self._registry = sam_model_registry
            self._Predictor = SamPredictor
            self._native = bool(checkpoint)
        except ImportError:
            pass

    def segment(self, req: SegmentRequest) -> SegmentResult:
        """画像をセグメンテーションする。"""
        if self._native and self.checkpoint:
            try:
                import numpy as np, io, torch
                from PIL import Image as PILImage  # type: ignore
                img = PILImage.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
                img_np = np.array(img)
                sam = self._registry[self.model_type](checkpoint=self.checkpoint)
                sam.to(device=self.device)
                predictor = self._Predictor(sam)
                predictor.set_image(img_np)
                input_points = np.array(req.points) if req.points else None
                input_labels = np.ones(len(req.points), dtype=int) if req.points else None
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=req.multimask,
                )
                result_masks = []
                for mask, score in zip(masks, scores):
                    buf = io.BytesIO()
                    PILImage.fromarray((mask * 255).astype("uint8")).save(buf, format="PNG")
                    result_masks.append(SegmentMask(
                        mask_b64=base64.b64encode(buf.getvalue()).decode(),
                        score=float(score),
                        area=int(mask.sum()),
                    ))
                return SegmentResult(masks=result_masks, n_masks=len(result_masks))
            except Exception:
                pass
        # fallback: ダミー
        PNG_1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        masks = [SegmentMask(mask_b64=PNG_1x1, score=0.9 - 0.1 * i, area=1000 - 100 * i)
                 for i in range(3 if req.multimask else 1)]
        return SegmentResult(masks=masks, n_masks=len(masks))

    @property
    def is_native(self) -> bool:
        return self._native
