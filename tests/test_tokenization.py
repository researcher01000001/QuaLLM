from src.cost_estimation.tokenization import count_tokens


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_counts_more_tokens_for_longer_text():
    short_count = count_tokens("hello")
    long_count = count_tokens("hello " * 50)

    assert long_count > short_count


def test_count_tokens_is_deterministic():
    text = "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung."

    assert count_tokens(text) == count_tokens(text)
