"""Unit tests for src.coding.ai_coding's condition-based tag filtering, and for
the AI-coding prompt snapshot feature"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.coding.ai_coding import (
    _extract_usage,
    prepare_document,
    process_documents,
    render_general_prompt,
    render_prompt,
    write_prompt_snapshots,
)
from src.coding.prompt_builders import DocumentPromptScheme, PromptContext
from src.config.coding import CodingConfiguration
from src.core.models import CodingScheme, Tag
from src.cost_estimation.tokenization import count_tokens

_EMPTY_CODING_SCHEME = CodingScheme(coding_scheme=[], general_rules=[], medical_knowledge={})


def test_process_documents_filters_tags_by_document_condition(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr(
        "src.coding.ai_coding.condition_match",
        lambda doc_name: ["general", "human"] if doc_name == "45_no advice_cycle_1" else [],
    )
    monkeypatch.setattr(
        "src.coding.ai_coding.load_prompt_context", lambda *a, **k: _EMPTY_CODING_SCHEME
    )

    doc_path = tmp_path / "45_no advice_cycle_1.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.", encoding="utf-8"
    )

    general_tag = Tag(name="General", condition="general")
    supported_tag = Tag(name="Supported", condition="supported")
    human_tag = Tag(name="Human", condition="human")

    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    captured_tags: list[list[Tag]] = []

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        captured_tags.append(tags)
        return []

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        process_documents(
            tmp_path,
            config,
            iteration=0,
            tags=[general_tag, supported_tag, human_tag],
        )

    assert len(captured_tags) == 1
    assert captured_tags[0] == [general_tag, human_tag]


def test_process_documents_keeps_only_unconditional_tags_for_unrecognised_document(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: [])
    monkeypatch.setattr(
        "src.coding.ai_coding.load_prompt_context", lambda *a, **k: _EMPTY_CODING_SCHEME
    )

    doc_path = tmp_path / "unknown_doc.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.", encoding="utf-8"
    )

    unconditional_tag = Tag(name="Unconditional")
    supported_tag = Tag(name="Supported", condition="supported")

    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    captured_tags: list[list[Tag]] = []

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        captured_tags.append(tags)
        return []

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        process_documents(
            tmp_path,
            config,
            iteration=0,
            tags=[unconditional_tag, supported_tag],
        )

    assert len(captured_tags) == 1
    assert captured_tags[0] == [unconditional_tag]


def test_process_documents_passes_scheme_and_row_to_analyse(tmp_path, monkeypatch):
    """process_documents() must thread the document's condition_match()
    result and raw condition_for() value into a DocumentPromptScheme,
    alongside the CodingScheme from load_prompt_context() -- this is what
    lets a prompt-builder function filter/format the coding scheme itself
    (see build_system_prompt_v1()). It must also resolve+pass each
    paragraph's real source row via resolve_rows_for_txt_file() (falling
    back to None here, since tmp_path isn't a recognized subfolder name)."""
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: ["general", "supported"])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: "advice")
    fake_scheme = CodingScheme(
        coding_scheme=[{"key": "row"}], general_rules=["rule"], medical_knowledge={}
    )
    monkeypatch.setattr("src.coding.ai_coding.load_prompt_context", lambda *a, **k: fake_scheme)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.", encoding="utf-8")

    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    captured = []

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        captured.append((row, scheme))
        return []

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        process_documents(tmp_path, config, iteration=0, tags=[])

    assert len(captured) == 1
    row, scheme = captured[0]
    assert row is None  # tmp_path isn't a recognized subfolder -- resolves to None, not a crash
    assert scheme.coding_scheme == fake_scheme
    assert scheme.valid_conditions == ["general", "supported"]
    assert scheme.document_condition == "advice"


def test_prepare_document_returns_none_for_missing_file():
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    result = prepare_document(
        Path("does/not/exist.txt"), config, tags=[], prompt_context=_EMPTY_CODING_SCHEME
    )

    assert result is None


def test_prepare_document_log_warnings_false_suppresses_missing_file_log(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.coding.ai_coding.process_log.current", lambda: mock_log)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    prepare_document(
        Path("does/not/exist.txt"),
        config,
        tags=[],
        prompt_context=_EMPTY_CODING_SCHEME,
        log_warnings=False,
    )

    mock_log.log_missing_source_document.assert_not_called()


def test_prepare_document_log_warnings_true_logs_missing_file(monkeypatch):
    mock_log = Mock()
    monkeypatch.setattr("src.coding.ai_coding.process_log.current", lambda: mock_log)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    prepare_document(
        Path("does/not/exist.txt"), config, tags=[], prompt_context=_EMPTY_CODING_SCHEME
    )

    mock_log.log_missing_source_document.assert_called_once()


def test_prepare_document_resolves_paragraphs_rows_tags_and_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: ["general"])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: "advice")

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.", encoding="utf-8"
    )
    general_tag = Tag(name="General", condition="general")
    supported_tag = Tag(name="Supported", condition="supported")
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    prep = prepare_document(
        doc_path, config, [general_tag, supported_tag], _EMPTY_CODING_SCHEME
    )

    assert prep is not None
    assert prep.filepath == doc_path
    assert prep.paragraphs == ["Ein Testabsatz mit genug Zeichen fuer die Verarbeitung."]
    assert prep.rows == [None]  # tmp_path isn't a recognized subfolder
    assert prep.tags == [general_tag]  # condition-filtered
    assert prep.scheme.valid_conditions == ["general"]
    assert prep.scheme.document_condition == "advice"


def test_prepare_document_resolves_rows_before_applying_the_paragraph_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.condition_match", lambda doc_name: ["general"])
    monkeypatch.setattr("src.coding.ai_coding.condition_for", lambda doc_name: "advice")
    monkeypatch.setattr("src.coding.ai_coding.AI_CODING_MAX_PARAGRAPHS", 2)
    seen_paragraphs = []

    def fake_resolve(filepath, paragraphs):
        seen_paragraphs.append(list(paragraphs))
        return [{"position": i} for i in range(len(paragraphs))]

    monkeypatch.setattr("src.coding.ai_coding.resolve_rows_for_txt_file", fake_resolve)
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("erster Absatz\nzweiter Absatz\ndritter Absatz\n", encoding="utf-8")
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    prep = prepare_document(doc_path, config, [], _EMPTY_CODING_SCHEME)

    assert prep is not None
    assert seen_paragraphs == [["erster Absatz", "zweiter Absatz", "dritter Absatz"]]
    assert prep.paragraphs == ["erster Absatz", "zweiter Absatz"]
    assert prep.rows == [{"position": 0}, {"position": 1}]


def test_extract_usage_reads_prompt_completion_cost_reasoning_and_paragraph_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cost=0.001,
            cache_discount=0.0004,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=10,
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
    )

    usage = _extract_usage(response, "Ein Testabsatz mit genug Zeichen.")

    assert usage is not None
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 120
    assert usage["cost"] == 0.001
    assert usage["cache_discount"] == 0.0004
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 80,
        "cache_write_tokens": 10,
    }
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 5}
    # paragraph_tokens is computed locally (not part of the API response) --
    # it's the basis calibration uses for completion_token_factor, so it must
    # be present and derived from the actual paragraph text passed in.
    assert usage["paragraph_tokens"] == count_tokens("Ein Testabsatz mit genug Zeichen.")
    assert usage["paragraph_tokens"] > 0


def test_extract_usage_returns_none_without_usage_field():
    assert _extract_usage(SimpleNamespace(), "some paragraph") is None


def test_process_documents_returns_none_when_cost_estimation_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr(
        "src.coding.ai_coding.load_prompt_context", lambda *a, **k: _EMPTY_CODING_SCHEME
    )
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Ein Testabsatz mit genug Zeichen fuer die Verarbeitung.", encoding="utf-8"
    )
    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        return [{"phrase": "x", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}]

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        result = process_documents(tmp_path, config, iteration=0, tags=[])

    assert result is None


def test_process_documents_aggregates_actual_usage_when_cost_estimation_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr(
        "src.coding.ai_coding.load_prompt_context", lambda *a, **k: _EMPTY_CODING_SCHEME
    )
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", True)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Erster Testabsatz mit genug Zeichen.\nZweiter Testabsatz mit genug Zeichen.",
        encoding="utf-8",
    )
    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        usage = {
            "prompt_tokens": 10,
            "paragraph_tokens": 5,
            "completion_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1},
            "prompt_tokens_details": {
                "cached_tokens": 7,
                "cache_write_tokens": 3,
            },
            "cache_discount": 0.004,
            "cost": 0.01,
        }
        source = {"sourcefile": filepath, "row": i}
        # One API call returned two classifications carrying the same legacy
        # call-level usage. Aggregation must count that usage only once.
        return [
            {"phrase": "x", "source": source, "usage": usage},
            {"phrase": "y", "source": source, "usage": usage},
        ]

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        result = process_documents(tmp_path, config, iteration=0, tags=[])

    assert result == {
        "prompt_tokens": 20,
        "paragraph_tokens": 10,
        "completion_tokens": 4,
        "reasoning_tokens": 2,
        "cached_prompt_tokens": 14,
        "cache_write_tokens": 6,
        "cache_discount": 0.008,
        "cost": 0.02,
        "calls": 2,
    }


def test_analyse_with_genai_requests_usage_accounting_when_cost_estimation_enabled(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", True)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost=0.0,
                completion_tokens_details=None,
            ),
            model="test/model",
            provider="test-provider",
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    assert captured_kwargs["extra_body"] == {"usage": {"include": True}}


def test_analyse_with_genai_attaches_call_usage_only_once(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", True)

    def fake_create(**kwargs):
        message = SimpleNamespace(
            content='{"classifications": [{"phrase": "x"}, {"phrase": "y"}]}'
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                cost=0.01,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=8, cache_write_tokens=0
                ),
                completion_tokens_details=None,
            ),
            model="test/model",
            provider="test-provider",
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    from src.coding.ai_coding import analyse_with_genai

    results = analyse_with_genai(
        0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, []
    )

    assert len(results) == 2
    assert "usage" in results[0]
    assert "usage" not in results[1]
    assert results[0]["usage"]["cache_discount"] is None


def test_analyse_with_genai_marks_system_prompt_for_caching_when_enabled(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], model="test/model", provider=None
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(
        short_name="test-config",
        model_name="test/model",
        prompt_caching=True,
    )

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    system_message = captured_kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert len(system_message["content"]) == 1
    assert system_message["content"][0]["type"] == "text"
    assert system_message["content"][0]["text"]
    assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured_kwargs["messages"][1]["role"] == "user"
    assert captured_kwargs["extra_body"] is None


def test_analyse_with_genai_keeps_plain_system_message_when_caching_disabled(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], model="test/model", provider=None
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(
        short_name="test-config",
        model_name="test/model",
        prompt_caching=False,
    )

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    assert isinstance(captured_kwargs["messages"][0]["content"], str)


def test_analyse_with_genai_omits_usage_accounting_when_cost_estimation_disabled(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], model="test/model", provider=None
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    assert captured_kwargs["extra_body"] is None


def test_analyse_with_genai_requests_reasoning_with_effective_effort_when_enabled(monkeypatch):
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], model="test/model", provider=None
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(short_name="test-config", model_name="test/model", reasoning=True)

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    assert captured_kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}


def test_analyse_with_genai_omits_reasoning_key_entirely_when_disabled(monkeypatch):
    """reasoning=False (the default) must not send "reasoning" at all --
    not as {"enabled": False}, simply absent."""
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", False)
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        message = SimpleNamespace(content='{"classifications": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], model="test/model", provider=None
        )

    monkeypatch.setattr("src.coding.ai_coding.client.chat.completions.create", fake_create)
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    from src.coding.ai_coding import analyse_with_genai

    analyse_with_genai(0, "Ein Testabsatz mit genug Zeichen.", 1, "doc.txt", config, [])

    assert captured_kwargs["extra_body"] is None


def test_process_documents_warm_start_skips_resubmitting_that_paragraph(tmp_path, monkeypatch):
    """A warm-started paragraph must not be sent to analyse_with_genai again
    -- its already-computed result is spliced in as-is and its usage is
    still counted exactly once."""
    monkeypatch.setattr("src.coding.ai_coding.get_ai_iteration_dir", lambda *a, **k: tmp_path / "output")
    monkeypatch.setattr(
        "src.coding.ai_coding.load_prompt_context", lambda *a, **k: _EMPTY_CODING_SCHEME
    )
    monkeypatch.setattr("src.coding.ai_coding.COST_ESTIMATION_ENABLED", True)

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Erster Testabsatz mit genug Zeichen.\nZweiter Testabsatz mit genug Zeichen.",
        encoding="utf-8",
    )
    config = CodingConfiguration(short_name="test-config", model_name="test/model", max_threads=1)

    called_indices: list[int] = []

    def fake_analyse(i, paragraph, total, filepath, coding_config, tags, row=None, scheme=None):
        called_indices.append(i)
        return [
            {
                "phrase": "y",
                "usage": {
                    "prompt_tokens": 10,
                    "paragraph_tokens": 5,
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "cost": 0.01,
                },
            }
        ]

    warm_start_result = [
        {
            "phrase": "x",
            "usage": {
                "prompt_tokens": 8,
                "paragraph_tokens": 4,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 1},
                "cost": 0.02,
            },
        }
    ]

    with patch("src.coding.ai_coding.analyse_with_genai", side_effect=fake_analyse):
        result = process_documents(
            tmp_path,
            config,
            iteration=0,
            tags=[],
            warm_start=(doc_path, 0, warm_start_result),
        )

    # Paragraph 0 (the warm-started one) was never resubmitted -- only
    # paragraph 1 went through analyse_with_genai.
    assert called_indices == [1]
    # Both the warm-started usage and the real call's usage are counted,
    # exactly once each.
    assert result is not None
    assert result["calls"] == 2
    assert result["completion_tokens"] == 5  # 3 (warm-started) + 2 (real call)
    assert result["cost"] == 0.03
    assert result["cache_discount"] is None


def test_render_general_prompt_resolves_tag_descriptions_but_placeholders_paragraph():
    """build_system_prompt_v0() (the default builder) reads tag descriptions
    from ctx.coding_scheme, not the raw tags list -- so a real CodingScheme
    must be supplied via scheme for them to show up."""
    config = CodingConfiguration(short_name="test-config", model_name="test/model")
    tags = [Tag(name="Navigationshandlung", description="Concrete movement description.")]
    coding_scheme = CodingScheme(
        coding_scheme=[
            {"key": "Navigationshandlung", "definition": "Concrete movement description."}
        ],
        general_rules=[],
        medical_knowledge={},
    )
    scheme = DocumentPromptScheme(
        coding_scheme=coding_scheme, valid_conditions=None, document_condition=None
    )

    system, user = render_general_prompt(config, tags, scheme)

    assert "Navigationshandlung" in system
    assert "Concrete movement description." in system
    assert "[paragraph text" in user


def test_render_prompt_defaults_to_empty_coding_scheme_without_scheme():
    """render_prompt() without a scheme falls back to an empty CodingScheme
    (see coding/ai_coding.py's _EMPTY_CODING_SCHEME) rather than erroring -- the
    default builder just has nothing to list."""
    config = CodingConfiguration(short_name="test-config", model_name="test/model")

    system, user = render_prompt(config, [], "some paragraph")

    assert "some paragraph" in user
    assert "Here are the ALLOWED TAGS:\n\n" in system


def test_render_prompt_makes_scheme_and_row_available_to_registered_builders(monkeypatch):
    """render_prompt() must build a PromptContext carrying the coding
    scheme, valid_conditions, document_condition, paragraph and row, and
    hand it to whichever builder the configuration names -- proven here by
    registering throwaway builders under a fake name and reading every
    field back out of the PromptContext they receive."""
    import src.coding.prompt_builders as prompt_builders_module

    captured_contexts: list[PromptContext] = []

    def fake_system_builder(ctx: PromptContext) -> str:
        captured_contexts.append(ctx)
        entry = ctx.coding_scheme.coding_scheme[0]
        return f"{entry['key']}={entry['definition']} conditions={ctx.valid_conditions} doc_condition={ctx.document_condition}"

    def fake_user_builder(ctx: PromptContext) -> str:
        return f"* {ctx.coding_scheme.general_rules[0]} row={ctx.row} paragraph={ctx.paragraph}"

    monkeypatch.setitem(
        prompt_builders_module.SYSTEM_PROMPT_BUILDERS, "_TEST_SYSTEM_BUILDER", fake_system_builder
    )
    monkeypatch.setitem(
        prompt_builders_module.USER_PROMPT_BUILDERS, "_TEST_USER_BUILDER", fake_user_builder
    )

    config = CodingConfiguration(
        short_name="test-config",
        model_name="test/model",
        system_prompt_name="_TEST_SYSTEM_BUILDER",
        user_prompt_name="_TEST_USER_BUILDER",
    )
    scheme = CodingScheme(
        coding_scheme=[{"key": "Foo", "definition": "Bar"}],
        general_rules=["Rule one."],
        medical_knowledge={},
    )
    document_scheme = DocumentPromptScheme(
        coding_scheme=scheme, valid_conditions=["general"], document_condition="advice"
    )

    system, user = render_prompt(
        config, [], "paragraph text", row={"id": 1}, scheme=document_scheme
    )

    assert system == "Foo=Bar conditions=['general'] doc_condition=advice"
    assert user == "* Rule one. row={'id': 1} paragraph=paragraph text"
    assert captured_contexts[0].row == {"id": 1}


def test_write_prompt_snapshots_writes_one_section_per_distinct_prompt(tmp_path):
    config_a = CodingConfiguration(short_name="config-a", model_name="test/model")
    config_b = CodingConfiguration(short_name="config-b", model_name="test/model-2")

    output_paths = write_prompt_snapshots(
        [
            ("advice", config_a, "system A", "user A"),
            ("advice", config_b, "system B", "user B"),
        ],
        output_dir=tmp_path,
    )

    assert output_paths == [tmp_path / "prompt_advice.txt"]
    content = output_paths[0].read_text(encoding="utf-8")
    assert "Configuration(s): config-a" in content
    assert "Configuration(s): config-b" in content
    assert "system A" in content
    assert "system B" in content


def test_write_prompt_snapshots_merges_configurations_with_identical_prompts(tmp_path):
    config_a = CodingConfiguration(short_name="config-a", model_name="test/model")
    config_b = CodingConfiguration(short_name="config-b", model_name="test/model")

    output_paths = write_prompt_snapshots(
        [
            ("advice", config_a, "same system", "same user"),
            ("advice", config_b, "same system", "same user"),
        ],
        output_dir=tmp_path,
    )

    content = output_paths[0].read_text(encoding="utf-8")
    assert "Configuration(s): config-a, config-b" in content
    # The prompt content itself must appear exactly once, not duplicated.
    assert content.count("same system") == 1


def test_write_prompt_snapshots_overwrites_existing_file(tmp_path):
    config = CodingConfiguration(short_name="config-a", model_name="test/model")
    write_prompt_snapshots(
        [("advice", config, "old system", "old user")], output_dir=tmp_path
    )

    output_paths = write_prompt_snapshots(
        [("advice", config, "new system", "new user")], output_dir=tmp_path
    )

    content = output_paths[0].read_text(encoding="utf-8")
    assert "old system" not in content
    assert "new system" in content


def test_write_prompt_snapshots_writes_one_file_per_condition(tmp_path):
    config = CodingConfiguration(short_name="config-a", model_name="test/model")

    output_paths = write_prompt_snapshots(
        [
            ("advice", config, "ALPHA system", "ALPHA user"),
            ("no advice", config, "BETA system", "BETA user"),
        ],
        output_dir=tmp_path,
    )

    assert output_paths == [
        tmp_path / "prompt_advice.txt",
        tmp_path / "prompt_no-advice.txt",
    ]
    advice_content = output_paths[0].read_text(encoding="utf-8")
    no_advice_content = output_paths[1].read_text(encoding="utf-8")
    assert "ALPHA system" in advice_content
    assert "BETA system" not in advice_content
    assert "BETA system" in no_advice_content
    assert "ALPHA system" not in no_advice_content


def test_write_prompt_snapshots_marks_conditions_absent_from_active_conditions(tmp_path):
    config = CodingConfiguration(short_name="config-a", model_name="test/model")

    output_paths = write_prompt_snapshots(
        [
            ("advice", config, "system", "user"),
            ("no advice", config, "system", "user"),
        ],
        active_conditions={"advice"},
        output_dir=tmp_path,
    )

    assert output_paths == [
        tmp_path / "prompt_advice.txt",
        tmp_path / "prompt_no-advice__unused.txt",
    ]


def test_write_prompt_snapshots_marks_nothing_when_active_conditions_omitted(tmp_path):
    config = CodingConfiguration(short_name="config-a", model_name="test/model")

    output_paths = write_prompt_snapshots(
        [("advice", config, "system", "user")],
        output_dir=tmp_path,
    )

    assert output_paths == [tmp_path / "prompt_advice.txt"]
