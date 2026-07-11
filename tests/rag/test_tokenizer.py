from govgrant.rag.index.hybrid import code_aware_tokenizer


def test_code_tokens():
    tokens = code_aware_tokenizer("See SF-424 and 2 CFR 200 under FOA-2024-01")
    joined = " ".join(tokens)
    assert "sf-424" in joined
    assert "foa-2024-01" in joined
