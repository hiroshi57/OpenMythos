"""
Sprint 61 — 日本語対応トークナイザー テスト (80 tests)

対象:
  open_mythos/tokenizer_ja.py:
    CharType / classify_char
    JaTokenizerConfig
    JaVocab
    JaSentenceSplitter
    JaTokenizer
    JaTokenizerAdapter
    build_vocab_from_corpus
"""
from __future__ import annotations

from open_mythos.tokenizer_ja import (
    CharType, classify_char,
    JaTokenizerConfig,
    JaVocab,
    JaSentenceSplitter,
    JaTokenizer,
    JaTokenizerAdapter,
    build_vocab_from_corpus,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CharType / classify_char
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCharType:
    def test_values(self):
        assert CharType.HIRAGANA.value == "hiragana"
        assert CharType.KATAKANA.value == "katakana"
        assert CharType.KANJI.value    == "kanji"
        assert CharType.ASCII.value    == "ascii"
        assert CharType.DIGIT.value    == "digit"
        assert CharType.PUNCT.value    == "punct"
        assert CharType.SPACE.value    == "space"
        assert CharType.OTHER.value    == "other"


class TestClassifyChar:
    def test_hiragana(self):
        assert classify_char("あ") == CharType.HIRAGANA
        assert classify_char("ん") == CharType.HIRAGANA

    def test_katakana(self):
        assert classify_char("ア") == CharType.KATAKANA
        assert classify_char("ー") == CharType.KATAKANA

    def test_kanji(self):
        assert classify_char("東") == CharType.KANJI
        assert classify_char("語") == CharType.KANJI

    def test_ascii(self):
        assert classify_char("a") == CharType.ASCII
        assert classify_char("Z") == CharType.ASCII

    def test_digit(self):
        assert classify_char("0") == CharType.DIGIT
        assert classify_char("9") == CharType.DIGIT

    def test_space(self):
        assert classify_char(" ") == CharType.SPACE
        assert classify_char("\t") == CharType.SPACE
        assert classify_char("\n") == CharType.SPACE

    def test_punct(self):
        assert classify_char("。") == CharType.PUNCT
        assert classify_char("、") == CharType.PUNCT
        assert classify_char("!") != CharType.ASCII  # ! は記号


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaTokenizerConfig
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaTokenizerConfig:
    def test_defaults(self):
        cfg = JaTokenizerConfig()
        assert cfg.max_length == 512
        assert cfg.pad_token  == "<pad>"
        assert cfg.unk_token  == "<unk>"
        assert cfg.bos_token  == "<bos>"
        assert cfg.eos_token  == "<eos>"
        assert cfg.kanji_ngram == 2

    def test_custom(self):
        cfg = JaTokenizerConfig(max_length=128, kanji_ngram=3)
        assert cfg.max_length  == 128
        assert cfg.kanji_ngram == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaVocab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaVocab:
    def test_special_tokens_present(self):
        vocab = JaVocab()
        assert "<pad>" in vocab.to_dict()
        assert "<unk>" in vocab.to_dict()
        assert "<bos>" in vocab.to_dict()
        assert "<eos>" in vocab.to_dict()

    def test_pad_id_is_0(self):
        vocab = JaVocab()
        assert vocab.pad_id == 0

    def test_unk_id_is_1(self):
        vocab = JaVocab()
        assert vocab.unk_id == 1

    def test_add_token_returns_id(self):
        vocab = JaVocab()
        i = vocab.add_token("東京")
        assert isinstance(i, int)
        assert i >= 4  # special tokens は 0-3

    def test_add_same_token_returns_same_id(self):
        vocab = JaVocab()
        i1 = vocab.add_token("東京")
        i2 = vocab.add_token("東京")
        assert i1 == i2

    def test_token_to_id_unknown(self):
        vocab = JaVocab()
        assert vocab.token_to_id("未登録") == vocab.unk_id

    def test_id_to_token_unknown(self):
        vocab = JaVocab()
        assert vocab.id_to_token(9999) == "<unk>"

    def test_vocab_size_grows(self):
        vocab = JaVocab()
        initial = vocab.vocab_size
        vocab.add_token("新トークン")
        assert vocab.vocab_size == initial + 1

    def test_tokens_list(self):
        vocab = JaVocab(tokens=["東京", "大阪"])
        assert "東京" in vocab.tokens()
        assert "大阪" in vocab.tokens()

    def test_most_frequent(self):
        vocab = JaVocab()
        vocab.add_token("東京")
        vocab.add_token("東京")
        vocab.add_token("大阪")
        top = vocab.most_frequent(1)
        assert top[0][0] == "東京"
        assert top[0][1] == 2

    def test_roundtrip(self):
        vocab = JaVocab()
        tid = vocab.add_token("テスト")
        assert vocab.id_to_token(tid) == "テスト"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaSentenceSplitter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaSentenceSplitter:
    def setup_method(self):
        self.sp = JaSentenceSplitter()

    def test_split_by_kuten(self):
        sents = self.sp.split("東京は晴れです。大阪は雨です。")
        assert len(sents) == 2
        assert "東京は晴れです。" in sents[0]

    def test_split_by_exclamation(self):
        sents = self.sp.split("すごい！本当にすごい！")
        assert len(sents) == 2

    def test_split_by_question(self):
        sents = self.sp.split("本当ですか？そうですか？")
        assert len(sents) == 2

    def test_single_sentence(self):
        sents = self.sp.split("こんにちは")
        assert len(sents) == 1

    def test_empty_string(self):
        sents = self.sp.split("")
        assert sents == []

    def test_split_into_chunks_short(self):
        chunks = self.sp.split_into_chunks("短い文。", max_chars=100)
        assert len(chunks) == 1

    def test_split_into_chunks_long(self):
        long_sent = "あ" * 200
        chunks = self.sp.split_into_chunks(long_sent, max_chars=100)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100

    def test_newline_splits(self):
        sents = self.sp.split("一行目\n二行目")
        assert len(sents) >= 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaTokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaTokenizer:
    def setup_method(self):
        self.tok = JaTokenizer()

    def test_tokenize_returns_list(self):
        tokens = self.tok.tokenize("東京タワー")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_kanji_split(self):
        # 漢字が N-gram に分割される
        tokens = self.tok.tokenize("東京観光")
        assert len(tokens) > 1  # 2gram で分割

    def test_tokenize_hiragana_grouped(self):
        tokens = self.tok.tokenize("あいうえお")
        assert "あいうえお" in tokens  # ひらがな連続はひとまとめ

    def test_tokenize_mixed(self):
        tokens = self.tok.tokenize("東京に行く")
        assert len(tokens) >= 2  # 漢字部分とひらがな部分

    def test_tokenize_empty(self):
        tokens = self.tok.tokenize("")
        assert tokens == []

    def test_encode_returns_ints(self):
        ids = self.tok.encode("こんにちは")
        assert all(isinstance(i, int) for i in ids)

    def test_encode_non_empty(self):
        ids = self.tok.encode("東京タワー")
        assert len(ids) > 0

    def test_encode_with_special_tokens(self):
        ids = self.tok.encode("テスト", add_special_tokens=True)
        assert ids[0] == self.tok.vocab.bos_id
        assert ids[-1] == self.tok.vocab.eos_id

    def test_encode_max_length_truncation(self):
        long_text = "あ" * 600
        ids = self.tok.encode(long_text, max_length=100)
        assert len(ids) <= 100

    def test_encode_padding(self):
        ids = self.tok.encode("短い", padding=True, max_length=20)
        assert len(ids) == 20
        # パディング部分は pad_id
        assert ids[-1] == self.tok.vocab.pad_id

    def test_decode_roundtrip(self):
        text = "東京タワーに行きたい"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_decode_hiragana_roundtrip(self):
        text = "こんにちは"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_decode_skip_special(self):
        ids = self.tok.encode("テスト", add_special_tokens=True)
        decoded = self.tok.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_encode_batch_same_length(self):
        texts = ["東京", "大阪観光", "京都"]
        batch = self.tok.encode_batch(texts, padding=True)
        lengths = [len(b) for b in batch]
        assert len(set(lengths)) == 1  # 全て同じ長さ

    def test_encode_batch_no_padding(self):
        texts = ["東京", "大阪観光"]
        batch = self.tok.encode_batch(texts, padding=False)
        assert len(batch[0]) != len(batch[1])  # 長さが異なる

    def test_tokenize_and_count(self):
        counts = self.tok.tokenize_and_count("東京東京大阪")
        assert isinstance(counts, dict)
        # 頻出トークンがカウントされる
        total = sum(counts.values())
        assert total > 0

    def test_char_type_map(self):
        # "東a1あア。" は 6 文字
        result = self.tok.char_type_map("東a1あア。")
        assert len(result) == 6
        assert result[0][1] == CharType.KANJI
        assert result[1][1] == CharType.ASCII
        assert result[2][1] == CharType.DIGIT
        assert result[3][1] == CharType.HIRAGANA
        assert result[4][1] == CharType.KATAKANA

    def test_vocab_grows_on_encode(self):
        tok = JaTokenizer()
        initial = tok.vocab_size
        tok.encode("全く新しいテキスト内容ですよ")
        assert tok.vocab_size > initial

    def test_ascii_lowercase(self):
        tok = JaTokenizer(config=JaTokenizerConfig(lowercase_ascii=True))
        tokens = tok.tokenize("Hello")
        assert all(t == t.lower() for t in tokens)

    def test_kanji_ngram_3(self):
        tok = JaTokenizer(config=JaTokenizerConfig(kanji_ngram=3))
        tokens = tok.tokenize("東京観光名所")
        # 3-gram で分割
        for t in tokens:
            ct = classify_char(t[0]) if t else CharType.OTHER
            if ct == CharType.KANJI:
                assert len(t) <= 3

    def test_no_kanji_split(self):
        tok = JaTokenizer(config=JaTokenizerConfig(split_kanji=False))
        tokens = tok.tokenize("東京観光")
        # 分割なしなので漢字連続がそのまま
        assert "東京観光" in tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaTokenizerAdapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaTokenizerAdapter:
    def setup_method(self):
        self.adapter = JaTokenizerAdapter()

    def test_encode_returns_ints(self):
        ids = self.adapter.encode("東京タワー")
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_str(self):
        ids = self.adapter.encode("東京タワー")
        text = self.adapter.decode(ids)
        assert isinstance(text, str)

    def test_vocab_size_positive(self):
        self.adapter.encode("東京大阪京都")
        assert self.adapter.vocab_size > 4

    def test_encode_decode_roundtrip(self):
        # 漢字 N-gram は重複があるため、ひらがな/カタカナのみでラウンドトリップを検証
        text = "こんにちはテスト"
        ids = self.adapter.encode(text)
        decoded = self.adapter.decode(ids)
        assert decoded == text

    def test_encode_with_special_tokens(self):
        ids = self.adapter.encode("テスト", add_special_tokens=True)
        assert len(ids) > 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# build_vocab_from_corpus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildVocabFromCorpus:
    def test_returns_ja_vocab(self):
        vocab = build_vocab_from_corpus(["東京タワー", "大阪城"])
        assert isinstance(vocab, JaVocab)

    def test_vocab_contains_tokens(self):
        vocab = build_vocab_from_corpus(["東京観光"])
        assert vocab.vocab_size > 4  # special tokens + 実トークン

    def test_min_freq_filter(self):
        texts = ["東京東京東京", "大阪"]  # 東京 3回, 大阪 1回
        vocab2 = build_vocab_from_corpus(texts, min_freq=2)
        vocab1 = build_vocab_from_corpus(texts, min_freq=1)
        # min_freq=2 の方が語彙数が少ない
        assert vocab2.vocab_size <= vocab1.vocab_size

    def test_empty_corpus(self):
        vocab = build_vocab_from_corpus([])
        assert vocab.vocab_size == 4  # special tokens のみ

    def test_freq_info_available(self):
        vocab = build_vocab_from_corpus(["東京東京大阪"])
        top = vocab.most_frequent(1)
        assert len(top) > 0

    def test_custom_config(self):
        cfg = JaTokenizerConfig(kanji_ngram=3)
        vocab = build_vocab_from_corpus(["東京観光名所"], config=cfg)
        assert vocab.vocab_size > 4
