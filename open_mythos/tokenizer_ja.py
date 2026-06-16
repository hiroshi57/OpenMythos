"""
Sprint 61 — 日本語対応トークナイザー

GPT-2 英語依存を解消し、日本語テキスト（広告コピー・CEP シナリオ等）を
外部ライブラリなしで処理できる軽量トークナイザー。

オブジェクト:
  CharType          : 文字種分類 (Hiragana/Katakana/Kanji/Ascii/Punct/Other)
  JaTokenizerConfig : 設定 (max_length / special tokens / pad / unk)
  JaVocab           : 語彙管理 (token → id / id → token / 頻度カウント)
  JaSentenceSplitter: 文分割 (句点・感嘆符・疑問符ベース)
  JaTokenizer       : メイントークナイザー (文字 N-gram + 文字種境界)
  JaTokenizerAdapter: MythosTokenizer 互換ラッパー (既存コードとの接続)
  build_vocab_from_corpus: コーパスから語彙を構築するユーティリティ

設計方針:
  - 外部依存なし (re / unicodedata のみ)
  - 文字種境界でトークン分割 (漢字→ひらがな等の切れ目)
  - 語彙外トークンは <unk> で処理
  - MythosTokenizer と同一 I/F (encode / decode / vocab_size)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文字種分類
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CharType(str, Enum):
    HIRAGANA = "hiragana"   # ひらがな (U+3041–U+309F)
    KATAKANA = "katakana"   # カタカナ (U+30A0–U+30FF)
    KANJI    = "kanji"      # 漢字 (U+4E00–U+9FFF + 拡張)
    ASCII    = "ascii"      # ASCII 文字 (半角英数)
    DIGIT    = "digit"      # 数字 (全角含む)
    PUNCT    = "punct"      # 句読点・記号
    SPACE    = "space"      # 空白
    OTHER    = "other"      # その他


def classify_char(ch: str) -> CharType:
    """1 文字の種別を返す"""
    cp = ord(ch)
    if 0x3041 <= cp <= 0x309F:
        return CharType.HIRAGANA
    if 0x30A0 <= cp <= 0x30FF:
        return CharType.KATAKANA
    if (0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF):
        return CharType.KANJI
    if ch.isdigit() or (0xFF10 <= cp <= 0xFF19):  # 全角数字
        return CharType.DIGIT
    if ch.isascii() and ch.isalnum():
        return CharType.ASCII
    if ch in " \t\n\r　":
        return CharType.SPACE
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return CharType.PUNCT
    return CharType.OTHER


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class JaTokenizerConfig:
    """トークナイザー設定"""
    max_length:        int  = 512
    pad_token:         str  = "<pad>"
    unk_token:         str  = "<unk>"
    bos_token:         str  = "<bos>"
    eos_token:         str  = "<eos>"
    split_kanji:       bool = True   # 漢字連続をN-gramで分割するか
    kanji_ngram:       int  = 2      # 漢字 N-gram サイズ
    lowercase_ascii:   bool = True   # ASCII を小文字化するか
    normalize_neologd: bool = True   # 全角→半角等の正規化


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 語彙管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class JaVocab:
    """
    語彙管理クラス。

    token → id と id → token の双方向マッピング。
    special tokens は先頭 4 ID に固定。
    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        config: Optional[JaTokenizerConfig] = None,
    ) -> None:
        self._config = config or JaTokenizerConfig()
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._freq: Dict[str, int] = {}

        # special tokens を登録
        for st in _SPECIAL_TOKENS:
            self._add(st)

        if tokens:
            for t in tokens:
                self._add(t)

    def _add(self, token: str) -> int:
        if token not in self._token2id:
            idx = len(self._token2id)
            self._token2id[token] = idx
            self._id2token[idx] = token
        return self._token2id[token]

    def add_token(self, token: str) -> int:
        """トークンを追加して ID を返す"""
        self._freq[token] = self._freq.get(token, 0) + 1
        return self._add(token)

    def token_to_id(self, token: str) -> int:
        """トークン → ID (未知は unk_id)"""
        return self._token2id.get(token, self._token2id.get(self._config.unk_token, 1))

    def id_to_token(self, token_id: int) -> str:
        """ID → トークン (未知は unk_token)"""
        return self._id2token.get(token_id, self._config.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self._token2id)

    @property
    def pad_id(self) -> int:
        return self._token2id.get(self._config.pad_token, 0)

    @property
    def unk_id(self) -> int:
        return self._token2id.get(self._config.unk_token, 1)

    @property
    def bos_id(self) -> int:
        return self._token2id.get(self._config.bos_token, 2)

    @property
    def eos_id(self) -> int:
        return self._token2id.get(self._config.eos_token, 3)

    def tokens(self) -> List[str]:
        return list(self._token2id.keys())

    def most_frequent(self, n: int = 10) -> List[Tuple[str, int]]:
        """頻度上位 n トークンを返す"""
        return sorted(self._freq.items(), key=lambda x: -x[1])[:n]

    def to_dict(self) -> Dict[str, int]:
        return dict(self._token2id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文分割
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JaSentenceSplitter:
    """
    日本語テキストを文単位に分割する。

    句点（。）・感嘆符（！）・疑問符（？）・改行で分割。
    """

    # 文末記号
    _SENT_END = re.compile(r"([。！？\n]+)")

    def split(self, text: str) -> List[str]:
        """テキストを文リストに分割する"""
        parts = self._SENT_END.split(text)
        sentences: List[str] = []
        buf = ""
        for i, part in enumerate(parts):
            if self._SENT_END.fullmatch(part):
                buf += part
                s = buf.strip()
                if s:
                    sentences.append(s)
                buf = ""
            else:
                buf += part
        if buf.strip():
            sentences.append(buf.strip())
        return sentences

    def split_into_chunks(self, text: str, max_chars: int = 100) -> List[str]:
        """文分割したあと max_chars を超える文をさらに分割する"""
        sents = self.split(text)
        chunks: List[str] = []
        for s in sents:
            if len(s) <= max_chars:
                chunks.append(s)
            else:
                # 単純に max_chars で切る
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i + max_chars])
        return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaTokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize(text: str, config: JaTokenizerConfig) -> str:
    """テキスト正規化（全角→半角、大文字→小文字）"""
    if not config.normalize_neologd:
        return text
    # Unicode NFKC 正規化（全角英数→半角、合字展開等）
    text = unicodedata.normalize("NFKC", text)
    if config.lowercase_ascii:
        # ASCII 部分のみ小文字化
        text = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), text)
    return text


def _split_by_char_type(text: str) -> List[str]:
    """
    文字種境界でテキストをセグメントに分割する。

    例: "東京タワー123abc" → ["東京", "タワー", "123", "abc"]
    """
    if not text:
        return []
    segments: List[str] = []
    buf = text[0]
    prev_type = classify_char(text[0])

    for ch in text[1:]:
        ct = classify_char(ch)
        if ct == prev_type or ct == CharType.SPACE or prev_type == CharType.SPACE:
            buf += ch
        else:
            if buf.strip():
                segments.append(buf)
            buf = ch
        prev_type = ct

    if buf.strip():
        segments.append(buf)
    return segments


def _kanji_ngrams(kanji_str: str, n: int) -> List[str]:
    """
    漢字文字列を N-gram に分割する。

    例: n=2, "東京タワー" → ["東京", "京タ", "タワ", "ワー"]
    ただし文字種分割後に呼ばれるので、漢字のみの文字列が渡る。
    len <= n の場合はそのまま返す。
    """
    if len(kanji_str) <= n:
        return [kanji_str]
    return [kanji_str[i:i + n] for i in range(len(kanji_str) - n + 1)]


class JaTokenizer:
    """
    日本語テキストをトークン列に分割するトークナイザー。

    分割アルゴリズム:
    1. テキスト正規化 (NFKC / lowercase)
    2. 文字種境界でセグメント化
    3. 漢字連続は N-gram に展開 (split_kanji=True 時)
    4. 語彙に登録してIDに変換

    Usage:
        tok = JaTokenizer()
        tokens = tok.tokenize("東京タワーに行きたい")
        ids = tok.encode("東京タワーに行きたい")
        text = tok.decode(ids)
    """

    def __init__(
        self,
        config: Optional[JaTokenizerConfig] = None,
        vocab: Optional[JaVocab] = None,
    ) -> None:
        self.config = config or JaTokenizerConfig()
        self.vocab  = vocab  or JaVocab(config=self.config)

    # ---- 公開 API (MythosTokenizer 互換) ----

    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size

    def tokenize(self, text: str) -> List[str]:
        """テキスト → トークン文字列リスト"""
        text = _normalize(text, self.config)
        segments = _split_by_char_type(text)
        tokens: List[str] = []
        for seg in segments:
            ct = classify_char(seg[0]) if seg else CharType.OTHER
            if (ct == CharType.KANJI
                    and self.config.split_kanji
                    and len(seg) > self.config.kanji_ngram):
                tokens.extend(_kanji_ngrams(seg, self.config.kanji_ngram))
            else:
                tokens.append(seg)
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        padding: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """テキスト → ID リスト"""
        tokens = self.tokenize(text)
        ids = [self.vocab.add_token(t) for t in tokens]
        if add_special_tokens:
            ids = [self.vocab.bos_id] + ids + [self.vocab.eos_id]
        ml = max_length or self.config.max_length
        ids = ids[:ml]
        if padding and len(ids) < ml:
            ids = ids + [self.vocab.pad_id] * (ml - len(ids))
        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """ID リスト → テキスト"""
        special_ids = {self.vocab.pad_id, self.vocab.bos_id, self.vocab.eos_id}
        tokens: List[str] = []
        for tid in token_ids:
            tok = self.vocab.id_to_token(tid)
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def encode_batch(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """複数テキストを一括エンコード（パディング対応）"""
        ml = max_length or self.config.max_length
        encoded = [
            self.encode(t, padding=False, max_length=ml) for t in texts
        ]
        if padding:
            max_len = max(len(e) for e in encoded) if encoded else 0
            encoded = [
                e + [self.vocab.pad_id] * (max_len - len(e))
                for e in encoded
            ]
        return encoded

    def tokenize_and_count(self, text: str) -> Dict[str, int]:
        """テキスト中のトークン頻度を返す"""
        tokens = self.tokenize(text)
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    def char_type_map(self, text: str) -> List[Tuple[str, CharType]]:
        """各文字とその文字種をペアで返す（デバッグ用）"""
        return [(ch, classify_char(ch)) for ch in text]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MythosTokenizer 互換ラッパー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JaTokenizerAdapter:
    """
    JaTokenizer を MythosTokenizer 互換 I/F でラップする。

    既存コードの `MythosTokenizer` を差し替えて使用できる。

    Usage:
        from open_mythos.tokenizer_ja import JaTokenizerAdapter
        tok = JaTokenizerAdapter()
        ids = tok.encode("東京タワーに行きたい")
        text = tok.decode(ids)
        n = tok.vocab_size
    """

    def __init__(self, config: Optional[JaTokenizerConfig] = None) -> None:
        self._tok = JaTokenizer(config=config)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_vocab_from_corpus(
    texts: List[str],
    config: Optional[JaTokenizerConfig] = None,
    min_freq: int = 1,
) -> JaVocab:
    """
    コーパスからトークン頻度をカウントし JaVocab を構築する。

    Args:
        texts    : テキストのリスト
        config   : トークナイザー設定
        min_freq : 語彙に含める最低頻度 (デフォルト 1 = 全トークン)

    Returns:
        頻度フィルタ適用済みの JaVocab
    """
    cfg = config or JaTokenizerConfig()
    tok = JaTokenizer(config=cfg)

    # 全テキストをトークン化して頻度カウント
    freq: Dict[str, int] = {}
    for text in texts:
        for t in tok.tokenize(text):
            freq[t] = freq.get(t, 0) + 1

    # min_freq フィルタ
    filtered = [t for t, cnt in freq.items() if cnt >= min_freq]

    vocab = JaVocab(tokens=filtered, config=cfg)
    # 頻度情報を vocab に転写
    for t, cnt in freq.items():
        if t in vocab._token2id:
            vocab._freq[t] = cnt

    return vocab


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 63C — 日本語形態素解析強化（辞書ベース分割）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PartOfSpeech(str, Enum):
    """品詞分類（簡易版）"""
    NOUN      = "noun"        # 名詞
    VERB      = "verb"        # 動詞
    ADJECTIVE = "adjective"   # 形容詞
    ADVERB    = "adverb"      # 副詞
    PARTICLE  = "particle"    # 助詞
    AUXILIARY = "auxiliary"   # 助動詞
    PREFIX    = "prefix"      # 接頭辞
    SUFFIX    = "suffix"      # 接尾辞
    SYMBOL    = "symbol"      # 記号
    UNKNOWN   = "unknown"     # 未知語


@dataclass
class DictionaryEntry:
    """辞書エントリ 1 件"""
    surface: str                          # 表層形
    pos:     PartOfSpeech = PartOfSpeech.NOUN
    reading: Optional[str] = None         # 読み（カタカナ）
    cost:    int           = 0            # 連接コスト（小さいほど優先）

    def to_dict(self) -> Dict[str, object]:
        return {
            "surface": self.surface,
            "pos":     self.pos.value,
            "reading": self.reading,
            "cost":    self.cost,
        }


@dataclass
class Morpheme:
    """形態素 1 件（解析結果）"""
    surface: str
    pos:     PartOfSpeech = PartOfSpeech.UNKNOWN
    reading: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "surface": self.surface,
            "pos":     self.pos.value,
            "reading": self.reading,
        }


# よく使う助詞・助動詞（最長一致用の最小辞書）
_DEFAULT_PARTICLES = [
    "について", "という", "ため", "から", "まで", "より", "など", "でも",
    "には", "では", "への", "との", "って", "けど", "のに", "ので",
    "は", "が", "を", "に", "へ", "と", "で", "も", "の", "や", "か",
    "ね", "よ", "な", "ば", "し",
]
_DEFAULT_AUXILIARY = [
    "ました", "ません", "ています", "ている", "たい", "だった", "です",
    "ます", "ない", "れる", "られる", "せる", "させる", "だ", "た",
]


class JaDictionary:
    """
    形態素解析用の辞書。

    表層形 → DictionaryEntry のマッピングを保持し、最長一致探索を支援する。
    デフォルトで基本的な助詞・助動詞を登録する。
    """

    def __init__(self, load_defaults: bool = True) -> None:
        self._entries: Dict[str, DictionaryEntry] = {}
        self._max_len = 0
        if load_defaults:
            self._load_defaults()

    def _load_defaults(self) -> None:
        for p in _DEFAULT_PARTICLES:
            self.add(DictionaryEntry(surface=p, pos=PartOfSpeech.PARTICLE))
        for a in _DEFAULT_AUXILIARY:
            self.add(DictionaryEntry(surface=a, pos=PartOfSpeech.AUXILIARY))

    def add(self, entry: DictionaryEntry) -> None:
        """辞書エントリを追加する"""
        self._entries[entry.surface] = entry
        self._max_len = max(self._max_len, len(entry.surface))

    def add_word(
        self,
        surface: str,
        pos: PartOfSpeech = PartOfSpeech.NOUN,
        reading: Optional[str] = None,
    ) -> None:
        """単語を辞書に追加するショートカット"""
        self.add(DictionaryEntry(surface=surface, pos=pos, reading=reading))

    def lookup(self, surface: str) -> Optional[DictionaryEntry]:
        """表層形でエントリを検索する"""
        return self._entries.get(surface)

    def contains(self, surface: str) -> bool:
        return surface in self._entries

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def max_word_length(self) -> int:
        return self._max_len


class JaMorphologicalAnalyzer:
    """
    辞書ベースの最長一致法による形態素解析器。

    アルゴリズム:
    1. 各位置から辞書の最長一致を探す
    2. マッチしたら Morpheme として切り出す
    3. マッチしなければ文字種境界まで未知語として切り出す

    Usage:
        analyzer = JaMorphologicalAnalyzer()
        analyzer.dictionary.add_word("東京", PartOfSpeech.NOUN, "トウキョウ")
        morphemes = analyzer.analyze("東京に行く")
    """

    def __init__(self, dictionary: Optional[JaDictionary] = None) -> None:
        self.dictionary = dictionary or JaDictionary()

    def analyze(self, text: str) -> List[Morpheme]:
        """テキストを形態素リストに分解する"""
        morphemes: List[Morpheme] = []
        i = 0
        n = len(text)
        max_len = max(1, self.dictionary.max_word_length)

        while i < n:
            # 空白はスキップ
            if classify_char(text[i]) == CharType.SPACE:
                i += 1
                continue

            matched = self._longest_match(text, i, max_len)
            if matched is not None:
                entry = self.dictionary.lookup(matched)
                morphemes.append(Morpheme(
                    surface=matched,
                    pos=entry.pos if entry else PartOfSpeech.UNKNOWN,
                    reading=entry.reading if entry else None,
                ))
                i += len(matched)
            else:
                # 未知語: 同一文字種が続く範囲を 1 形態素にする
                seg = self._unknown_segment(text, i)
                pos = self._guess_pos(seg)
                morphemes.append(Morpheme(surface=seg, pos=pos))
                i += len(seg)

        return morphemes

    def _longest_match(self, text: str, start: int, max_len: int) -> Optional[str]:
        """start 位置から辞書の最長一致を探す"""
        end = min(len(text), start + max_len)
        for length in range(end - start, 0, -1):
            candidate = text[start:start + length]
            if self.dictionary.contains(candidate):
                return candidate
        return None

    def _unknown_segment(self, text: str, start: int) -> str:
        """
        未知語セグメントを切り出す。
        同一文字種が続く範囲を取得するが、辞書にヒットする位置で止める。
        """
        i = start
        n = len(text)
        first_type = classify_char(text[start])
        buf = text[start]
        i += 1
        while i < n:
            ct = classify_char(text[i])
            if ct != first_type:
                break
            # 途中で辞書語が始まるなら止める
            if self.dictionary.contains(text[i]):
                break
            buf += text[i]
            i += 1
        return buf

    def _guess_pos(self, surface: str) -> PartOfSpeech:
        """未知語の品詞を文字種から推測する"""
        if not surface:
            return PartOfSpeech.UNKNOWN
        ct = classify_char(surface[0])
        if ct == CharType.KANJI or ct == CharType.KATAKANA:
            return PartOfSpeech.NOUN
        if ct == CharType.PUNCT:
            return PartOfSpeech.SYMBOL
        return PartOfSpeech.UNKNOWN

    def surfaces(self, text: str) -> List[str]:
        """形態素の表層形のみのリストを返す"""
        return [m.surface for m in self.analyze(text)]

    def filter_by_pos(self, text: str, pos: PartOfSpeech) -> List[Morpheme]:
        """指定品詞の形態素のみを返す"""
        return [m for m in self.analyze(text) if m.pos == pos]

    def extract_nouns(self, text: str) -> List[str]:
        """名詞の表層形を抽出する（広告キーワード抽出に有用）"""
        return [m.surface for m in self.filter_by_pos(text, PartOfSpeech.NOUN)]
