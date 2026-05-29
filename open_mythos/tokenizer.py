DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> "MythosTokenizer":
    """Convenience factory — create and return a :class:`MythosTokenizer`.

    Args:
        model_id: HuggingFace model identifier or local path.
                  Defaults to ``"openai/gpt-oss-20b"``.

    Returns:
        A ready-to-use :class:`MythosTokenizer` instance.

    Example:
        >>> tok = load_tokenizer()
        >>> tok.encode("hello")
        [15339]
    """
    return MythosTokenizer(model_id=model_id)


def get_vocab_size(model_id: str = DEFAULT_MODEL_ID) -> int:
    """Return the vocabulary size for the given tokenizer without keeping it alive.

    Args:
        model_id: HuggingFace model identifier or local path.
                  Defaults to ``"openai/gpt-oss-20b"``.

    Returns:
        Integer vocabulary size.

    Example:
        >>> n = get_vocab_size()
        >>> assert n > 0
    """
    return MythosTokenizer(model_id=model_id).vocab_size


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
        """
        from transformers import AutoTokenizer  # lazy import to avoid mock contamination
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
