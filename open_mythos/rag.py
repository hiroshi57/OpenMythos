"""
OpenMythos RAG (Retrieval-Augmented Generation) Pipeline。

Claude の knowledge retrieval に対応するオープン実装。
numpy cosine similarity ベースの軽量ベクターストアと、
FAISS のオプション対応を備える。

設計:
    Document        -- ドキュメントと埋め込みベクトルのコンテナ
    VectorStore     -- numpy (デフォルト) / FAISS (オプション) 検索エンジン
    RAGPipeline     -- ドキュメント追加・検索・生成の一気通貫パイプライン

使い方::

    from open_mythos.rag import RAGPipeline

    pipeline = RAGPipeline(model, device="cpu")
    pipeline.add_documents([
        "LLMOとは AI検索エンジン向けのコンテンツ最適化手法です。",
        "CTRはクリック率。広告効果の基本指標。",
    ])
    result = pipeline.generate_with_context(
        query="LLMOの定義を教えて",
        top_k=2,
        max_new_tokens=80,
    )
    print(result.answer)
    print(result.retrieved_docs)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from open_mythos.main import OpenMythos

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """ドキュメントと埋め込みベクトルのコンテナ。"""

    text: str
    """ドキュメント本文。"""

    embedding: Optional[torch.Tensor] = None
    """埋め込みベクトル (encode 後に設定される)。shape: (embed_dim,)"""

    doc_id: str = ""
    """ドキュメントID。省略時は自動生成。"""

    metadata: dict = field(default_factory=dict)
    """任意のメタデータ (source URL、カテゴリ等)。"""

    score: float = 0.0
    """検索スコア (retrieve 後に設定)。"""


@dataclass
class RAGResult:
    """RAG生成結果。"""

    query: str
    """入力クエリ。"""

    answer: str
    """生成された回答。"""

    retrieved_docs: list[Document]
    """検索されたドキュメントのリスト。"""

    context_used: str
    """モデルに渡されたコンテキスト文字列。"""

    latency_ms: float = 0.0
    """生成にかかった時間 (ms)。"""

    n_docs_in_store: int = 0
    """ベクターストア内の総ドキュメント数。"""


# ---------------------------------------------------------------------------
# 埋め込みエンコーダ (軽量: TF-IDF 類似の bag-of-chars)
# ---------------------------------------------------------------------------


class _BagOfCharsEncoder:
    """
    ライブラリ依存なし、軽量な文字レベル Bag-of-Chars 埋め込み。

    外部 embedding モデルが不要なため CPU のみでも動作する。
    本番では sentence-transformers などに差し替えることを推奨。
    """

    VOCAB_SIZE = 256  # ASCII + extended

    def encode(self, text: str, dim: int = 256) -> torch.Tensor:
        """
        テキストを固定長ベクトルに変換する。

        文字ユニグラム頻度 + 位置重み付き TF を計算し、
        dim 次元に射影する。
        """
        n = len(text)
        if n == 0:
            return torch.zeros(dim)

        # 文字頻度ベクトル (256 次元)
        counts = torch.zeros(self.VOCAB_SIZE)
        for i, c in enumerate(text):
            pos_weight = 1.0 + 0.5 * math.exp(-i / max(n, 1))  # 前方重み
            counts[ord(c) % self.VOCAB_SIZE] += pos_weight

        counts = counts / (counts.norm() + 1e-8)

        # dim > VOCAB_SIZE の場合: バイグラム成分を追加
        if dim > self.VOCAB_SIZE:
            bigrams = torch.zeros(self.VOCAB_SIZE)
            for i in range(n - 1):
                bigrams[(ord(text[i]) + ord(text[i + 1])) % self.VOCAB_SIZE] += 1.0
            bigrams = bigrams / (bigrams.norm() + 1e-8)
            combined = torch.cat([counts, bigrams])
            # dim に合わせてリピートまたはトランケート
            if combined.shape[0] < dim:
                reps = math.ceil(dim / combined.shape[0])
                combined = combined.repeat(reps)[:dim]
            else:
                combined = combined[:dim]
            return combined / (combined.norm() + 1e-8)

        return counts[:dim]

    def encode_batch(self, texts: list[str], dim: int = 256) -> torch.Tensor:
        """複数テキストを一括エンコードする。shape: (n, dim)"""
        return torch.stack([self.encode(t, dim) for t in texts])


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """
    ドキュメント埋め込みの保存と類似度検索。

    numpy (デフォルト) または FAISS (インストール済みの場合) を使用。
    """

    def __init__(self, embed_dim: int = 256, use_faiss: bool = False) -> None:
        self.embed_dim = embed_dim
        self._docs: list[Document] = []
        self._embeddings: Optional[torch.Tensor] = None  # (n, embed_dim)
        self._use_faiss = use_faiss
        self._faiss_index = None

        if use_faiss:
            try:
                import faiss  # type: ignore
                self._faiss_index = faiss.IndexFlatIP(embed_dim)  # Inner Product
                self._use_faiss = True
            except ImportError:
                self._use_faiss = False  # graceful fallback to numpy

    # ------------------------------------------------------------------
    # 追加
    # ------------------------------------------------------------------

    def add(self, docs: list[Document]) -> None:
        """ドキュメントを追加する。embedding が設定済みであること。"""
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document '{doc.doc_id}' has no embedding. Call encode() first.")
            if not doc.doc_id:
                doc.doc_id = f"doc_{len(self._docs)}"
            self._docs.append(doc)

        if docs:
            new_embs = torch.stack([d.embedding for d in docs])  # (n, dim)
            if self._embeddings is None:
                self._embeddings = new_embs
            else:
                self._embeddings = torch.cat([self._embeddings, new_embs], dim=0)

            if self._use_faiss and self._faiss_index is not None:
                import faiss
                emb_np = new_embs.float().numpy()
                faiss.normalize_L2(emb_np)
                self._faiss_index.add(emb_np)

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # 検索
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> list[Document]:
        """
        クエリ埋め込みに最も類似したドキュメントを返す。

        Args:
            query_embedding -- クエリベクトル (embed_dim,)
            top_k           -- 返すドキュメント数

        Returns:
            スコア降順の Document リスト (score フィールドに類似度)
        """
        if not self._docs:
            return []

        top_k = min(top_k, len(self._docs))

        if self._use_faiss and self._faiss_index is not None:
            import faiss

            q = query_embedding.float().unsqueeze(0).numpy()
            faiss.normalize_L2(q)
            scores, indices = self._faiss_index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = Document(
                    text=self._docs[idx].text,
                    doc_id=self._docs[idx].doc_id,
                    metadata=self._docs[idx].metadata,
                    score=float(score),
                )
                results.append(doc)
            return results

        # numpy cosine similarity
        assert self._embeddings is not None
        q = F.normalize(query_embedding.float().unsqueeze(0), dim=-1)  # (1, dim)
        embs = F.normalize(self._embeddings.float(), dim=-1)  # (n, dim)
        scores = (embs @ q.T).squeeze(-1)  # (n,)

        top_indices = scores.topk(top_k).indices.tolist()
        results = []
        for idx in top_indices:
            doc = Document(
                text=self._docs[idx].text,
                doc_id=self._docs[idx].doc_id,
                metadata=self._docs[idx].metadata,
                score=float(scores[idx]),
            )
            results.append(doc)
        return results


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Retrieval-Augmented Generation パイプライン。

    1. `add_documents()` でドキュメントをインデックス化
    2. `retrieve()` でクエリに類似したドキュメントを検索
    3. `generate_with_context()` で検索結果をコンテキストに含めて生成

    Args:
        model      -- OpenMythos モデルインスタンス
        device     -- torch device
        embed_dim  -- 埋め込み次元数
        use_faiss  -- FAISS インデックスを使う場合 True
    """

    def __init__(
        self,
        model: "OpenMythos",
        device: str = "cpu",
        embed_dim: int = 256,
        use_faiss: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self._encoder = _BagOfCharsEncoder()
        self._store = VectorStore(embed_dim=embed_dim, use_faiss=use_faiss)
        self._embed_dim = embed_dim

    # ------------------------------------------------------------------
    # ドキュメント管理
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[list[dict]] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> int:
        """
        テキストをエンコードしてインデックスに追加する。

        Args:
            texts     -- ドキュメントテキストのリスト
            metadatas -- 各ドキュメントのメタデータ (オプション)
            doc_ids   -- 各ドキュメントのID (オプション)

        Returns:
            追加したドキュメント数
        """
        docs: list[Document] = []
        for i, text in enumerate(texts):
            emb = self._encoder.encode(text, self._embed_dim)
            doc = Document(
                text=text,
                embedding=emb,
                doc_id=doc_ids[i] if doc_ids else f"doc_{len(self._store) + i}",
                metadata=metadatas[i] if metadatas else {},
            )
            docs.append(doc)

        self._store.add(docs)
        return len(docs)

    def n_docs(self) -> int:
        """インデックス内のドキュメント数を返す。"""
        return len(self._store)

    # ------------------------------------------------------------------
    # 検索
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """
        クエリに最も関連するドキュメントを検索する。

        Args:
            query -- 検索クエリ
            top_k -- 返すドキュメント数

        Returns:
            関連度降順の Document リスト
        """
        q_emb = self._encoder.encode(query, self._embed_dim)
        return self._store.search(q_emb, top_k=top_k)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def generate_with_context(
        self,
        query: str,
        top_k: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        loops: int = 4,
        context_prefix: str = "以下の参考情報を元に回答してください:\n",
    ) -> RAGResult:
        """
        関連ドキュメントを取得してコンテキストに組み込み、回答を生成する。

        Args:
            query           -- ユーザークエリ
            top_k           -- 検索するドキュメント数
            max_new_tokens  -- 最大生成トークン数
            temperature     -- サンプリング温度
            top_p           -- nucleus sampling 閾値
            loops           -- 推論ループ数
            context_prefix  -- コンテキストブロックの前置詞

        Returns:
            RAGResult
        """
        t0 = time.perf_counter()

        # 1. 検索
        retrieved = self.retrieve(query, top_k=top_k)

        # 2. コンテキスト構築
        if retrieved:
            context_parts = [context_prefix]
            for i, doc in enumerate(retrieved, 1):
                context_parts.append(f"[{i}] {doc.text}")
            context_parts.append(f"\n質問: {query}\n回答:")
            context = "\n".join(context_parts)
        else:
            context = f"質問: {query}\n回答:"

        # 3. トークナイズ & 生成
        vsize = self.model.cfg.vocab_size
        max_prompt = max(1, self.model.cfg.max_seq_len - max_new_tokens - 4)
        ids = [ord(c) % vsize for c in context[:max_prompt]]
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        generated_ids: list[int] = []
        cur_ids = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(cur_ids, n_loops=loops)
                next_logits = logits[0, -1, :] / max(temperature, 1e-8)

                if top_p < 1.0:
                    sorted_l, sorted_idx = torch.sort(next_logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                    remove = cum - F.softmax(sorted_l, dim=-1) > top_p
                    sorted_l[remove] = float("-inf")
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                        0, sorted_idx, sorted_l
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
                generated_ids.append(next_token)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )
                if next_token == vsize - 1:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000

        # 4. デコード
        chars = []
        for i in generated_ids:
            try:
                c = chr(i % 128)
                if c.isprintable() or c in "\n\t ":
                    chars.append(c)
            except (ValueError, OverflowError):
                pass
        answer = "".join(chars)

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_docs=retrieved,
            context_used=context,
            latency_ms=round(latency_ms, 2),
            n_docs_in_store=len(self._store),
        )
