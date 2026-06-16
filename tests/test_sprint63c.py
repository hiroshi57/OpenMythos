"""
Sprint 63C — 日本語形態素解析強化 テスト

対象:
  open_mythos/tokenizer_ja.py（追加分）:
    PartOfSpeech
    DictionaryEntry / Morpheme
    JaDictionary
    JaMorphologicalAnalyzer
"""
from __future__ import annotations

import pytest

from open_mythos.tokenizer_ja import (
    PartOfSpeech,
    DictionaryEntry, Morpheme,
    JaDictionary,
    JaMorphologicalAnalyzer,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PartOfSpeech
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPartOfSpeech:
    def test_values(self):
        assert PartOfSpeech.NOUN.value      == "noun"
        assert PartOfSpeech.VERB.value      == "verb"
        assert PartOfSpeech.PARTICLE.value  == "particle"
        assert PartOfSpeech.AUXILIARY.value == "auxiliary"
        assert PartOfSpeech.UNKNOWN.value   == "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DictionaryEntry / Morpheme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDictionaryEntry:
    def test_defaults(self):
        e = DictionaryEntry(surface="東京")
        assert e.pos == PartOfSpeech.NOUN
        assert e.reading is None

    def test_to_dict(self):
        e = DictionaryEntry(surface="東京", pos=PartOfSpeech.NOUN, reading="トウキョウ")
        d = e.to_dict()
        assert d["surface"] == "東京"
        assert d["pos"] == "noun"
        assert d["reading"] == "トウキョウ"


class TestMorpheme:
    def test_defaults(self):
        m = Morpheme(surface="走る")
        assert m.pos == PartOfSpeech.UNKNOWN

    def test_to_dict(self):
        m = Morpheme(surface="走る", pos=PartOfSpeech.VERB)
        d = m.to_dict()
        assert d["surface"] == "走る"
        assert d["pos"] == "verb"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaDictionary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaDictionary:
    def test_default_particles_loaded(self):
        d = JaDictionary()
        assert d.contains("は")
        assert d.contains("を")

    def test_default_auxiliary_loaded(self):
        d = JaDictionary()
        assert d.contains("です")
        assert d.contains("ます")

    def test_no_defaults(self):
        d = JaDictionary(load_defaults=False)
        assert d.size == 0

    def test_add_word(self):
        d = JaDictionary(load_defaults=False)
        d.add_word("東京", PartOfSpeech.NOUN, "トウキョウ")
        assert d.contains("東京")

    def test_lookup(self):
        d = JaDictionary(load_defaults=False)
        d.add_word("東京", PartOfSpeech.NOUN)
        entry = d.lookup("東京")
        assert entry.pos == PartOfSpeech.NOUN

    def test_lookup_missing(self):
        d = JaDictionary(load_defaults=False)
        assert d.lookup("nope") is None

    def test_size(self):
        d = JaDictionary(load_defaults=False)
        d.add_word("a")
        d.add_word("b")
        assert d.size == 2

    def test_max_word_length(self):
        d = JaDictionary(load_defaults=False)
        d.add_word("東")
        d.add_word("東京都")
        assert d.max_word_length == 3

    def test_add_entry(self):
        d = JaDictionary(load_defaults=False)
        d.add(DictionaryEntry(surface="走る", pos=PartOfSpeech.VERB))
        assert d.contains("走る")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JaMorphologicalAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJaMorphologicalAnalyzer:
    def setup_method(self):
        self.analyzer = JaMorphologicalAnalyzer()
        # テスト用の名詞・動詞を追加
        self.analyzer.dictionary.add_word("東京", PartOfSpeech.NOUN, "トウキョウ")
        self.analyzer.dictionary.add_word("大阪", PartOfSpeech.NOUN, "オオサカ")
        self.analyzer.dictionary.add_word("行く", PartOfSpeech.VERB, "イク")
        self.analyzer.dictionary.add_word("広告", PartOfSpeech.NOUN, "コウコク")

    def test_analyze_returns_morphemes(self):
        result = self.analyzer.analyze("東京に行く")
        assert all(isinstance(m, Morpheme) for m in result)
        assert len(result) > 0

    def test_analyze_dictionary_match(self):
        result = self.analyzer.analyze("東京")
        assert result[0].surface == "東京"
        assert result[0].pos == PartOfSpeech.NOUN

    def test_analyze_particle(self):
        result = self.analyzer.analyze("東京に")
        surfaces = [m.surface for m in result]
        assert "東京" in surfaces
        assert "に" in surfaces
        # 助詞として認識
        ni = [m for m in result if m.surface == "に"][0]
        assert ni.pos == PartOfSpeech.PARTICLE

    def test_analyze_longest_match(self):
        # 「東京」が「東」「京」に分かれず一語になる
        result = self.analyzer.analyze("東京")
        assert len(result) == 1

    def test_analyze_reading_preserved(self):
        result = self.analyzer.analyze("東京")
        assert result[0].reading == "トウキョウ"

    def test_analyze_unknown_word(self):
        result = self.analyzer.analyze("ＸＹＺ商品")
        # 未知語も何らかの形態素になる
        assert len(result) > 0

    def test_analyze_mixed(self):
        result = self.analyzer.analyze("東京と大阪に行く")
        surfaces = [m.surface for m in result]
        assert "東京" in surfaces
        assert "大阪" in surfaces
        assert "行く" in surfaces

    def test_analyze_empty(self):
        result = self.analyzer.analyze("")
        assert result == []

    def test_analyze_skips_spaces(self):
        result = self.analyzer.analyze("東京 大阪")
        surfaces = [m.surface for m in result]
        assert "東京" in surfaces
        assert "大阪" in surfaces
        # 空白は形態素にならない
        assert " " not in surfaces

    def test_surfaces(self):
        surfaces = self.analyzer.surfaces("東京に行く")
        assert isinstance(surfaces, list)
        assert "東京" in surfaces

    def test_filter_by_pos_noun(self):
        nouns = self.analyzer.filter_by_pos("東京と大阪", PartOfSpeech.NOUN)
        noun_surfaces = [m.surface for m in nouns]
        assert "東京" in noun_surfaces
        assert "大阪" in noun_surfaces

    def test_filter_by_pos_particle(self):
        particles = self.analyzer.filter_by_pos("東京と大阪", PartOfSpeech.PARTICLE)
        particle_surfaces = [m.surface for m in particles]
        assert "と" in particle_surfaces

    def test_extract_nouns(self):
        nouns = self.analyzer.extract_nouns("東京で広告を出す")
        assert "東京" in nouns
        assert "広告" in nouns

    def test_default_dictionary(self):
        analyzer = JaMorphologicalAnalyzer()
        # デフォルト辞書だけでも動作する
        result = analyzer.analyze("これはテストです")
        assert len(result) > 0

    def test_custom_dictionary(self):
        d = JaDictionary(load_defaults=False)
        d.add_word("特殊用語", PartOfSpeech.NOUN)
        analyzer = JaMorphologicalAnalyzer(dictionary=d)
        result = analyzer.analyze("特殊用語")
        assert result[0].surface == "特殊用語"

    def test_katakana_noun_guess(self):
        analyzer = JaMorphologicalAnalyzer(dictionary=JaDictionary(load_defaults=False))
        result = analyzer.analyze("コンピュータ")
        # カタカナ語は名詞推定
        assert result[0].pos == PartOfSpeech.NOUN
