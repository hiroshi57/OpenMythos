"""
Sprint 48 — マルチモーダル統合 テスト

対象:
  - open_mythos/skills/multimodal.py:
      CLIPEmbedding / CLIPModel
      VisionChatMessage / VisionChatResult / LLaVAModel
      DiffusionRequest / DiffusionResult / StableDiffusionGenerator
      SegmentRequest / SegmentMask / SegmentResult / SAMSegmenter
  - serve/api.py:
      POST /v1/clip/encode/text
      POST /v1/clip/encode/image
      POST /v1/clip/classify
      POST /v1/llava/chat
      POST /v1/diffusion/generate
      POST /v1/sam/segment
"""
from __future__ import annotations

import sys
import math
import base64
import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# transformers モック
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# TestClient フィクスチャ
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "ok " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32,
        act_threshold=0.99, lora_rank=4, kv_lora_rank=32, q_lora_rank=64,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )
    model = OpenMythos(cfg)
    model.eval()
    api_module.state.model = model
    api_module.state.tokenizer = tok
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())
    api_module.state.llm = OpenMythosLLM(model=model, device="cpu")
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


_HDR = {"Authorization": "Bearer dev"}

# 1x1 白 PNG (base64)
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

from open_mythos.skills.multimodal import (
    CLIPEmbedding, CLIPModel,
    VisionChatMessage, VisionChatResult, LLaVAModel,
    DiffusionRequest, DiffusionResult, StableDiffusionGenerator,
    SegmentRequest, SegmentMask, SegmentResult, SAMSegmenter,
)


# ---------------------------------------------------------------------------
# Section A: CLIPEmbedding / CLIPModel
# ---------------------------------------------------------------------------

class TestCLIPEmbedding:
    def test_creation(self):
        emb = CLIPEmbedding(vector=[0.1, 0.2, 0.3], modality="text", dim=3)
        assert emb.modality == "text"
        assert emb.dim == 3

    def test_similarity_self_is_one(self):
        emb = CLIPEmbedding(vector=[1.0, 0.0, 0.0], modality="text", dim=3)
        assert abs(emb.similarity(emb) - 1.0) < 1e-6

    def test_similarity_orthogonal_is_zero(self):
        a = CLIPEmbedding(vector=[1.0, 0.0], modality="text", dim=2)
        b = CLIPEmbedding(vector=[0.0, 1.0], modality="text", dim=2)
        assert abs(a.similarity(b)) < 1e-6

    def test_similarity_returns_float(self):
        a = CLIPEmbedding(vector=[0.5, 0.5], modality="image", dim=2)
        b = CLIPEmbedding(vector=[0.3, 0.7], modality="text", dim=2)
        assert isinstance(a.similarity(b), float)


class TestCLIPModel:
    def _make(self) -> CLIPModel:
        clip = CLIPModel()
        clip._native = False  # transformers がモック済み
        return clip

    def test_is_native_bool(self):
        clip = CLIPModel()
        assert isinstance(clip.is_native, bool)

    def test_encode_text_returns_list(self):
        clip = self._make()
        embs = clip.encode_text(["hello", "world"])
        assert len(embs) == 2

    def test_encode_text_type(self):
        clip = self._make()
        embs = clip.encode_text(["test"])
        assert isinstance(embs[0], CLIPEmbedding)

    def test_encode_text_modality(self):
        clip = self._make()
        embs = clip.encode_text(["ai research"])
        assert embs[0].modality == "text"

    def test_encode_text_dim_positive(self):
        clip = self._make()
        embs = clip.encode_text(["hello"])
        assert embs[0].dim > 0

    def test_encode_image_b64(self):
        clip = self._make()
        emb = clip.encode_image_b64(_PNG_B64)
        assert isinstance(emb, CLIPEmbedding)
        assert emb.modality == "image"

    def test_zero_shot_classify_returns_dict(self):
        clip = self._make()
        probs = clip.zero_shot_classify(_PNG_B64, ["cat", "dog", "car"])
        assert isinstance(probs, dict)
        assert set(probs.keys()) == {"cat", "dog", "car"}

    def test_zero_shot_classify_sums_to_one(self):
        clip = self._make()
        probs = clip.zero_shot_classify(_PNG_B64, ["A", "B", "C"])
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Section B: LLaVAModel
# ---------------------------------------------------------------------------

class TestVisionChatMessage:
    def test_creation(self):
        msg = VisionChatMessage(role="user", text="What is this?")
        assert msg.role == "user"
        assert msg.image_b64 is None

    def test_with_image(self):
        msg = VisionChatMessage(role="user", text="Describe", image_b64=_PNG_B64)
        assert msg.image_b64 == _PNG_B64


class TestLLaVAModel:
    def _make(self) -> LLaVAModel:
        m = LLaVAModel()
        m._native = False
        return m

    def test_is_native_bool(self):
        m = LLaVAModel()
        assert isinstance(m.is_native, bool)

    def test_chat_returns_result(self):
        m = self._make()
        msgs = [VisionChatMessage(role="user", text="Hello")]
        result = m.chat(msgs)
        assert isinstance(result, VisionChatResult)

    def test_chat_response_nonempty(self):
        m = self._make()
        msgs = [VisionChatMessage(role="user", text="Describe this image", image_b64=_PNG_B64)]
        result = m.chat(msgs)
        assert len(result.response) > 0

    def test_chat_tokens_used_nonneg(self):
        m = self._make()
        msgs = [VisionChatMessage(role="user", text="Hi")]
        result = m.chat(msgs)
        assert result.tokens_used >= 0


# ---------------------------------------------------------------------------
# Section C: StableDiffusionGenerator
# ---------------------------------------------------------------------------

class TestDiffusionRequest:
    def test_defaults(self):
        req = DiffusionRequest(prompt="a cat")
        assert req.width == 512
        assert req.height == 512
        assert req.steps == 20
        assert req.guidance_scale == 7.5

    def test_custom(self):
        req = DiffusionRequest(prompt="sunset", steps=30, width=768)
        assert req.steps == 30
        assert req.width == 768


class TestStableDiffusionGenerator:
    def _make(self) -> StableDiffusionGenerator:
        gen = StableDiffusionGenerator()
        gen._native = False
        return gen

    def test_is_native_bool(self):
        gen = StableDiffusionGenerator()
        assert isinstance(gen.is_native, bool)

    def test_generate_returns_result(self):
        gen = self._make()
        req = DiffusionRequest(prompt="a beautiful landscape")
        result = gen.generate(req)
        assert isinstance(result, DiffusionResult)

    def test_generate_image_b64_nonempty(self):
        gen = self._make()
        result = gen.generate(DiffusionRequest(prompt="test"))
        assert len(result.image_b64) > 0

    def test_generate_prompt_echoed(self):
        gen = self._make()
        result = gen.generate(DiffusionRequest(prompt="my prompt"))
        assert result.prompt == "my prompt"

    def test_generate_seed_assigned(self):
        gen = self._make()
        result = gen.generate(DiffusionRequest(prompt="x", seed=42))
        assert result.seed == 42


# ---------------------------------------------------------------------------
# Section D: SAMSegmenter
# ---------------------------------------------------------------------------

class TestSegmentRequest:
    def test_creation(self):
        req = SegmentRequest(image_b64=_PNG_B64, points=[(10, 20)])
        assert len(req.points) == 1
        assert req.multimask is True


class TestSAMSegmenter:
    def test_is_native_bool(self):
        sam = SAMSegmenter()
        assert isinstance(sam.is_native, bool)

    def test_segment_returns_result(self):
        sam = SAMSegmenter()
        req = SegmentRequest(image_b64=_PNG_B64)
        result = sam.segment(req)
        assert isinstance(result, SegmentResult)

    def test_segment_has_masks(self):
        sam = SAMSegmenter()
        result = sam.segment(SegmentRequest(image_b64=_PNG_B64))
        assert isinstance(result.masks, list)
        assert len(result.masks) > 0

    def test_segment_n_masks_matches(self):
        sam = SAMSegmenter()
        result = sam.segment(SegmentRequest(image_b64=_PNG_B64, multimask=True))
        assert result.n_masks == len(result.masks)

    def test_segment_mask_score_range(self):
        sam = SAMSegmenter()
        result = sam.segment(SegmentRequest(image_b64=_PNG_B64))
        for mask in result.masks:
            assert 0.0 <= mask.score <= 1.0

    def test_single_mask_mode(self):
        sam = SAMSegmenter()
        result = sam.segment(SegmentRequest(image_b64=_PNG_B64, multimask=False))
        assert result.n_masks == 1


# ---------------------------------------------------------------------------
# Section E: API エンドポイント
# ---------------------------------------------------------------------------

class TestCLIPEncodeTextEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/clip/encode/text",
                        json={"texts": ["hello world"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_embeddings(self, client):
        r = client.post("/v1/clip/encode/text",
                        json={"texts": ["ai", "ml"]},
                        headers=_HDR)
        assert isinstance(r.json()["embeddings"], list)
        assert len(r.json()["embeddings"]) == 2

    def test_embedding_has_vector(self, client):
        r = client.post("/v1/clip/encode/text",
                        json={"texts": ["test"]},
                        headers=_HDR)
        emb = r.json()["embeddings"][0]
        assert "vector" in emb
        assert isinstance(emb["vector"], list)


class TestCLIPEncodeImageEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/clip/encode/image",
                        json={"image_b64": _PNG_B64},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_embedding(self, client):
        r = client.post("/v1/clip/encode/image",
                        json={"image_b64": _PNG_B64},
                        headers=_HDR)
        assert "embedding" in r.json()
        assert r.json()["embedding"]["modality"] == "image"


class TestCLIPClassifyEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/clip/classify",
                        json={"image_b64": _PNG_B64, "labels": ["cat", "dog"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_scores(self, client):
        r = client.post("/v1/clip/classify",
                        json={"image_b64": _PNG_B64, "labels": ["a", "b", "c"]},
                        headers=_HDR)
        assert "scores" in r.json()
        scores = r.json()["scores"]
        assert set(scores.keys()) == {"a", "b", "c"}


class TestLLaVAChatEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/llava/chat",
                        json={"messages": [{"role": "user", "text": "Hello"}]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_response(self, client):
        r = client.post("/v1/llava/chat",
                        json={"messages": [{"role": "user", "text": "What is AI?"}]},
                        headers=_HDR)
        assert "response" in r.json()
        assert len(r.json()["response"]) > 0

    def test_with_image(self, client):
        r = client.post("/v1/llava/chat",
                        json={"messages": [{"role": "user", "text": "Describe", "image_b64": _PNG_B64}]},
                        headers=_HDR)
        assert r.status_code == 200


class TestDiffusionGenerateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/diffusion/generate",
                        json={"prompt": "a cat on a mat"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_image_b64(self, client):
        r = client.post("/v1/diffusion/generate",
                        json={"prompt": "sunset"},
                        headers=_HDR)
        assert "image_b64" in r.json()
        assert len(r.json()["image_b64"]) > 0

    def test_prompt_echoed(self, client):
        r = client.post("/v1/diffusion/generate",
                        json={"prompt": "my test prompt"},
                        headers=_HDR)
        assert r.json()["prompt"] == "my test prompt"


class TestSAMSegmentEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/sam/segment",
                        json={"image_b64": _PNG_B64},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_masks(self, client):
        r = client.post("/v1/sam/segment",
                        json={"image_b64": _PNG_B64},
                        headers=_HDR)
        assert isinstance(r.json()["masks"], list)

    def test_n_masks_positive(self, client):
        r = client.post("/v1/sam/segment",
                        json={"image_b64": _PNG_B64, "multimask": True},
                        headers=_HDR)
        assert r.json()["n_masks"] > 0
