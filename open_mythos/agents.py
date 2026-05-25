"""
OpenMythos Agent Integrations.

Provides drop-in adapters so OpenMythos models can be used inside
LangChain pipelines and Swarms multi-agent frameworks without
requiring either library to be installed.

Classes
-------
OpenMythosLLM
    LangChain ``BaseLLM`` compatible adapter.  Install with::

        pip install langchain-core

MythosAgent
    Swarms ``Agent`` compatible wrapper.  Install with::

        pip install swarms

Both classes fall back gracefully when the optional dependency is absent —
they still expose a ``run(prompt)`` method for direct use.

Usage
-----
LangChain::

    from open_mythos.agents import OpenMythosLLM
    llm = OpenMythosLLM.from_variant("nano", max_new_tokens=64)
    result = llm.invoke("Once upon a time")

Swarms::

    from open_mythos.agents import MythosAgent
    agent = MythosAgent.from_variant("nano", agent_name="narrator")
    response = agent.run("Write a story opening.")

Direct (no framework)::

    from open_mythos.agents import OpenMythosLLM
    llm = OpenMythosLLM.from_variant("nano")
    print(llm.run("Hello world"))
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

import torch

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.cli import _tokenize, _detokenize
from open_mythos.variants import (
    mythos_nano,
    mythos_1b,
    mythos_3b,
    mythos_7b,
    mythos_10b,
)

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}


def _load_model(
    variant: str = "nano",
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[OpenMythos, str]:
    """Load an OpenMythos model from a variant name or checkpoint path."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg: MythosConfig = ckpt["cfg"]
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        variant_fn = _VARIANTS.get(variant)
        if variant_fn is None:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}"
            )
        cfg = variant_fn()
        model = OpenMythos(cfg)
    return model.to(dev).eval(), dev


# ---------------------------------------------------------------------------
# Base mixin — framework-agnostic generate logic
# ---------------------------------------------------------------------------


class _MythosGenerateMixin:
    """Shared generation logic used by both LangChain and Swarms adapters."""

    _model: OpenMythos
    _device: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float

    def _generate_text(self, prompt: str) -> str:
        vocab_size = self._model.cfg.vocab_size
        ids = _tokenize(prompt, vocab_size)
        if not ids:
            ids = [0]  # BOS fallback for empty prompt
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        generated = out[0, input_ids.shape[1] :].tolist()
        return _detokenize(generated)

    def _stream_text(self, prompt: str) -> Iterator[str]:
        vocab_size = self._model.cfg.vocab_size
        ids = _tokenize(prompt, vocab_size)
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        for token_ids in self._model.generate_stream(**kwargs):
            yield _detokenize([token_ids[0, -1].item()])


# ---------------------------------------------------------------------------
# 6.3.1  OpenMythosLLM — LangChain BaseLLM adapter
# ---------------------------------------------------------------------------

try:
    from langchain_core.language_models.llms import BaseLLM

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseLLM = object  # type: ignore[assignment,misc]


class OpenMythosLLM(_MythosGenerateMixin, BaseLLM):  # type: ignore[misc]
    """LangChain ``BaseLLM`` adapter for OpenMythos.

    When ``langchain-core`` is not installed this class still works as a
    standalone LLM via ``run(prompt)`` and ``stream(prompt)`` — only the
    LangChain pipeline integration is unavailable.

    Args:
        model          -- pre-loaded ``OpenMythos`` instance
        device         -- torch device string the model lives on
        max_new_tokens -- tokens to generate per call
        temperature    -- sampling temperature (1.0 = unscaled)
        top_k          -- top-K filtering (0 = disabled)
        top_p          -- nucleus sampling threshold
    """

    # Pydantic fields (LangChain requires class-level annotations)
    if _HAS_LANGCHAIN:
        from pydantic import Field as _Field

        max_new_tokens: int = 128
        temperature: float = 1.0
        top_k: int = 50
        top_p: float = 0.9
    else:
        max_new_tokens: int = 128
        temperature: float = 1.0
        top_k: int = 50
        top_p: float = 0.9

    def __init__(
        self,
        model: OpenMythos,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> None:
        if _HAS_LANGCHAIN:
            super().__init__(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
        else:
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_k = top_k
            self.top_p = top_p
        # Store model outside Pydantic so it is never serialised
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_device", device)

    # ------------------------------------------------------------------
    # LangChain protocol
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "open-mythos"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "pip install langchain-core to use LangChain pipeline methods"
            )
        from langchain_core.outputs import Generation, LLMResult

        generations = []
        for prompt in prompts:
            text = self._generate_text(prompt)
            if stop:
                for s in stop:
                    if s in text:
                        text = text[: text.index(s)]
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "pip install langchain-core to use LangChain pipeline methods"
            )
        from langchain_core.outputs import GenerationChunk

        for chunk in self._stream_text(prompt):
            yield GenerationChunk(text=chunk)

    # ------------------------------------------------------------------
    # Standalone API (no LangChain required)
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> str:
        """Generate text from prompt without requiring LangChain."""
        return self._generate_text(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream tokens from prompt without requiring LangChain."""
        return self._stream_text(prompt)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        variant: str = "nano",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenMythosLLM":
        """Construct from a named variant or checkpoint.

        Args:
            variant    -- model size: ``"nano"``, ``"1b"``, ``"3b"``, etc.
            checkpoint -- path to a ``.pt`` checkpoint (overrides variant)
            device     -- torch device (auto-detects CUDA if None)
            **kwargs   -- forwarded to ``OpenMythosLLM.__init__``
        """
        model, dev = _load_model(variant, checkpoint, device)
        return cls(model=model, device=dev, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "OpenMythosLLM":
        """Load from a local path or Hugging Face Hub repo.

        Args:
            repo_id_or_path -- local ``.pt`` file / directory or HF repo ID
            device          -- torch device (auto-detects CUDA if None)
            **kwargs        -- forwarded to ``OpenMythosLLM.__init__``
        """
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = OpenMythos.from_pretrained(repo_id_or_path, map_location=dev)
        return cls(model=model, device=dev, **kwargs)


# ---------------------------------------------------------------------------
# 6.3.2  MythosAgent — Swarms Agent compatible wrapper
# ---------------------------------------------------------------------------

try:
    from swarms import Agent as SwarmsAgent  # type: ignore[import]

    _HAS_SWARMS = True
except ImportError:
    _HAS_SWARMS = False
    SwarmsAgent = object  # type: ignore[assignment,misc]


class MythosAgent(_MythosGenerateMixin):
    """Swarms ``Agent``-compatible wrapper for OpenMythos.

    Mirrors the ``Agent.run(task)`` interface so a ``MythosAgent`` can be
    dropped into any Swarms pipeline that expects an agent-like object.
    When ``swarms`` is not installed the class works as a standalone
    agent via ``run(task)``.

    Args:
        model          -- pre-loaded ``OpenMythos`` instance
        device         -- torch device string
        agent_name     -- display name (shown in multi-agent logs)
        system_prompt  -- optional system prefix prepended to every task
        max_new_tokens -- tokens to generate per ``run`` call
        temperature    -- sampling temperature
        top_k          -- top-K filtering
        top_p          -- nucleus sampling threshold
    """

    def __init__(
        self,
        model: OpenMythos,
        device: str = "cpu",
        agent_name: str = "MythosAgent",
        system_prompt: str = "",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        self._model = model
        self._device = device
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._history: list[str] = []

    # ------------------------------------------------------------------
    # Swarms / standalone run interface
    # ------------------------------------------------------------------

    def run(self, task: str, *args: Any, **kwargs: Any) -> str:
        """Generate a response for the given task.

        Prepends ``system_prompt`` when set, and maintains a conversation
        history so multi-turn tasks can accumulate context.

        Args:
            task -- the user instruction or question

        Returns:
            Generated response string.
        """
        prompt = self._build_prompt(task)
        response = self._generate_text(prompt)
        self._history.append(f"User: {task}\nAssistant: {response}")
        return response

    def stream_run(self, task: str) -> Iterator[str]:
        """Stream response tokens for the given task."""
        prompt = self._build_prompt(task)
        response_chunks: list[str] = []
        for chunk in self._stream_text(prompt):
            response_chunks.append(chunk)
            yield chunk
        self._history.append(f"User: {task}\nAssistant: {''.join(response_chunks)}")

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Swarms Agent protocol methods
    # ------------------------------------------------------------------

    def __call__(self, task: str, *args: Any, **kwargs: Any) -> str:
        return self.run(task, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"MythosAgent(name={self.agent_name!r}, "
            f"variant={self._model.cfg.dim}d, "
            f"max_new_tokens={self.max_new_tokens})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, task: str) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self._history:
            # Include last 2 turns for context (keeps prompt short on nano)
            parts.extend(self._history[-2:])
        parts.append(f"User: {task}\nAssistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        variant: str = "nano",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "MythosAgent":
        """Construct from a named variant or checkpoint."""
        model, dev = _load_model(variant, checkpoint, device)
        return cls(model=model, device=dev, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "MythosAgent":
        """Load from a local path or Hugging Face Hub repo."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = OpenMythos.from_pretrained(repo_id_or_path, map_location=dev)
        return cls(model=model, device=dev, **kwargs)
