"""Unit tests for data loaders"""

import json
from pathlib import Path

import pandas as pd

from src.config.coding import CodingConfiguration
from src.core import process_log
from src.core.models import AIQuote, ConsensusQuote, Document, Tag
from src.io.loaders import (
    AIQuoteLoader,
    AtlasQuoteLoader,
    ConsensusQuoteLoader,
    DocumentLoader,
    load_prompt_context,
    load_tags,
)


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_plain_text(self, tmp_txt_file: Path):
        loader = DocumentLoader(tmp_txt_file)
        doc = loader.load()

        assert isinstance(doc, Document)
        assert doc.source_name == "test_doc"
        assert len(doc.rows) == 2

        p0 = doc.rows[0]
        assert p0.index == 0
        assert p0.text == "The patient shows clear signs of improvement."
        assert p0.start_position == 0
        assert p0.end_position == len(p0.text)

        p1 = doc.rows[1]
        assert p1.index == 1
        assert p1.text == "However, further tests are needed."
        # +1 for the newline separating paragraphs
        assert p1.start_position == p0.end_position + 1

    def test_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        doc = DocumentLoader(empty).load()
        assert len(doc.rows) == 0

    def test_single_line_no_newline(self, tmp_path: Path):
        single = tmp_path / "single.txt"
        single.write_text("only line", encoding="utf-8")
        doc = DocumentLoader(single).load()
        assert len(doc.rows) == 1
        assert doc.rows[0].text == "only line"


class TestLoadTags:
    """Tests for load_tags with mock Excel files."""

    def test_load_from_codes_only(self, tmp_path: Path):
        """When only a codes file is provided, tags should use the color and a title-cased name."""
        import openpyxl
        from openpyxl.styles import Font

        codes_path = tmp_path / "codes.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.append(["Code Name"])
        cell = ws.cell(row=2, column=1, value="Monitoring")
        cell.font = Font(color="FFFF0000")
        wb.save(codes_path)

        tags = load_tags(codes_path)
        assert len(tags) == 1
        assert tags[0].name == "Monitoring"
        assert tags[0].color is not None
        assert tags[0].color.lower() == "#ff0000"

    def test_load_with_descriptions(self, tmp_path: Path):
        """When descriptions are provided, tag metadata should be fully populated."""
        import openpyxl
        from openpyxl.styles import Font

        codes_path = tmp_path / "codes.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.append(["Code Name"])
        cell = ws.cell(row=2, column=1, value="Monitoring")
        cell.font = Font(color="FF00FF00")
        wb.save(codes_path)

        desc_path = tmp_path / "descriptions.xlsx"
        import pandas as pd

        df = pd.DataFrame(
            {
                "name_1": ["Monitoring"],
                "name_2": ["Subcategory"],
                "group": ["Metacognition"],
                "definition": ["Tracking one's own understanding."],
                "note": ["Important note."],
                "examples": ['"example 1", "example 2"'],
                "condition": ["advice"],
            }
        )
        df.to_excel(desc_path, sheet_name="Coding_Scheme", index=False)

        tags = load_tags(codes_path, desc_path)
        assert len(tags) == 1
        tag = tags[0]
        assert tag.name == "Monitoring: Subcategory"
        assert tag.group == "Metacognition"
        assert tag.description == "Tracking one's own understanding."
        assert tag.note == "Important note."
        assert tag.examples == ["example 1", "example 2"]
        assert tag.condition == "advice"

    def test_system_column_sets_tag_system_flag(self, tmp_path: Path):
        """The coding scheme's ``system`` column (yes/no) becomes Tag.system;
        a missing value means not a system tag."""
        desc_path = tmp_path / "descriptions.xlsx"
        pd.DataFrame(
            {
                "code_category": ["other", "reasoning", "reflection"],
                "code": [None, "artifact", "self-monitoring"],
                "group": ["Other", "Cognition", "Self-Metacognition"],
                "system": ["yes", "no", None],
            }
        ).to_excel(desc_path, sheet_name="Coding_Scheme", index=False)

        tags = load_tags(tmp_path / "missing_codes.xlsx", desc_path)

        assert [(tag.name, tag.system) for tag in tags] == [
            ("other", True),
            ("reasoning: artifact", False),
            ("reflection: self-monitoring", False),
        ]

    def test_missing_codes_file_uses_name_derived_colors(self, tmp_path: Path):
        """A missing codes file falls back to the undefined-tag hash colour per tag."""
        import hashlib

        desc_path = tmp_path / "descriptions.xlsx"
        pd.DataFrame(
            {
                "name_1": ["Monitoring", "Planning"],
                "name_2": ["Subcategory", None],
                "group": ["Metacognition", "Metacognition"],
                "definition": ["Def 1", "Def 2"],
            }
        ).to_excel(desc_path, sheet_name="Coding_Scheme", index=False)

        tags = load_tags(tmp_path / "missing_codes.xlsx", desc_path)

        assert [tag.name for tag in tags] == ["Monitoring: Subcategory", "Planning"]
        for tag in tags:
            expected = f"#{hashlib.md5(tag.name.encode('utf-8')).hexdigest()[:6]}"
            assert tag.color == expected
        assert tags[0].color != tags[1].color


class TestLoadPromptContext:
    """load_prompt_context() gives AI-coding prompt construction a
    CodingScheme with raw, already-cleaned access to Coding_Scheme/
    General_Rules/Medical_Knowledge -- see src.coding.prompt_builders."""

    def _write_description_file(self, path: Path) -> None:
        coding_scheme = pd.DataFrame(
            {
                "code_category": ["Indicator", "Confidence"],
                # First row has a blank code/definition -- must clean to
                # ""/None, never the literal string "nan".
                "code": [None, "Sure"],
                "group": ["Group A", None],
                "definition": [None, "Explicit confidence statement."],
                "note": [None, None],
                "examples": [None, '"ex 1", "ex 2"'],
                "condition": ["general", "supported"],
                "double_coding": ["no", "yes"],
                "system": ["no", "no"],
            }
        )
        general_rules = pd.DataFrame({"number": [1, 2, 3], "rule": ["Rule one.", "Rule two.", None]})
        medical_knowledge = pd.DataFrame(
            {
                "indicators_of_healthy_anatomy": ["Symmetric ventricles", None],
                "Unnamed: 1": [None, None],
            }
        )
        with pd.ExcelWriter(path) as writer:
            coding_scheme.to_excel(writer, sheet_name="Coding_Scheme", index=False)
            general_rules.to_excel(writer, sheet_name="General_Rules", index=False)
            medical_knowledge.to_excel(writer, sheet_name="Medical_Knowledge", index=False)

    def test_coding_scheme_rows_are_cleaned(self, tmp_path: Path):
        desc_path = tmp_path / "descriptions.xlsx"
        self._write_description_file(desc_path)

        context = load_prompt_context(desc_path)

        rows = context.coding_scheme
        assert len(rows) == 2
        blank_row = rows[0]
        assert blank_row["key"] == "Indicator"
        assert blank_row["code_category"] == "Indicator"
        assert blank_row["code"] == ""
        assert blank_row["definition"] is None
        assert blank_row["examples"] == []
        assert blank_row["double_coding"] is False
        full_row = rows[1]
        assert full_row["key"] == "Confidence: Sure"
        assert full_row["code"] == "Sure"
        assert full_row["definition"] == "Explicit confidence statement."
        assert full_row["examples"] == ["ex 1", "ex 2"]
        assert full_row["double_coding"] is True
        assert full_row["condition"] == "supported"

    def test_general_rules_drops_blank_rows(self, tmp_path: Path):
        desc_path = tmp_path / "descriptions.xlsx"
        self._write_description_file(desc_path)

        context = load_prompt_context(desc_path)

        assert context.general_rules == ["Rule one.", "Rule two."]

    def test_medical_knowledge_skips_unnamed_columns_and_keeps_ragged_lists(
        self, tmp_path: Path
    ):
        desc_path = tmp_path / "descriptions.xlsx"
        self._write_description_file(desc_path)

        context = load_prompt_context(desc_path)

        assert context.medical_knowledge == {
            "indicators_of_healthy_anatomy": ["Symmetric ventricles"]
        }


class TestAIQuoteLoader:
    def test_load_runs_keeps_iterations_separate(self, tmp_path: Path, sample_tags):
        config = CodingConfiguration(short_name="test-config", model_name="test/model")
        for iteration in (0, 1):
            iteration_dir = (
                tmp_path
                / "prefix"
                / config.configuration_name
                / f"iteration-{iteration}"
            )
            iteration_dir.mkdir(parents=True)
            (iteration_dir / "doc_classified.json").write_text(
                json.dumps(
                    [
                        {
                            "phrase": "text",
                            "context": "surrounding text here",
                            "tags": ["Monitoring"],
                            "source": {
                                "sourcefile": "doc.txt",
                                "row": iteration,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

        runs = AIQuoteLoader.load_runs(
            base_dir=tmp_path / "prefix",
            configurations=[config],
            iterations=2,
            tags=sample_tags,
        )

        iter_0_quote = runs[(config.configuration_name, 0)]["doc"][0]
        iter_1_quote = runs[(config.configuration_name, 1)]["doc"][0]

        assert isinstance(iter_0_quote, AIQuote)
        assert isinstance(iter_1_quote, AIQuote)
        assert iter_0_quote.row == 0
        assert iter_0_quote.context == "surrounding text here"
        assert iter_1_quote.row == 1

    def test_load_runs_uses_configuration_iteration_counts(
        self,
        tmp_path: Path,
        sample_tags,
    ):
        config_one = CodingConfiguration(
            short_name="config-one",
            model_name="test/model",
            iterations=1,
        )
        config_two = CodingConfiguration(
            short_name="config-two",
            model_name="test/model",
            iterations=2,
        )

        for config, iteration in (
            (config_one, 0),
            (config_two, 0),
            (config_two, 1),
        ):
            iteration_dir = (
                tmp_path
                / "prefix"
                / config.configuration_name
                / f"iteration-{iteration}"
            )
            iteration_dir.mkdir(parents=True)
            (iteration_dir / "doc_classified.json").write_text(
                json.dumps(
                    [
                        {
                            "phrase": "text",
                            "tags": ["Monitoring"],
                            "source": {
                                "sourcefile": "doc.txt",
                                "row": iteration,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

        runs = AIQuoteLoader.load_runs(
            base_dir=tmp_path / "prefix",
            configurations=[config_one, config_two],
            tags=sample_tags,
        )

        assert set(runs) == {
            (config_one.configuration_name, 0),
            (config_two.configuration_name, 0),
            (config_two.configuration_name, 1),
        }


class TestAIQuoteLoaderUndefinedTags:
    """Tests for handling tag names absent from the codebook."""

    @staticmethod
    def _write_classified_json(directory: Path, tag_names: list[str]) -> None:
        (directory / "doc_classified.json").write_text(
            json.dumps(
                [
                    {
                        "phrase": f"phrase {i}",
                        "tags": [tag_name],
                        "source": {"sourcefile": "doc.txt", "row": i},
                    }
                    for i, tag_name in enumerate(tag_names)
                ]
            ),
            encoding="utf-8",
        )

    def test_undefined_tags_kept_by_default(self, tmp_path: Path, sample_tags):
        """UNDEFINED_TAGS_ENABLED defaults True (src/config/general.py) -- an
        undefined tag is kept, marked additional_tag=True, not dropped. See
        test_undefined_tag_logs_to_process_log below for the dropped case
        (UNDEFINED_TAGS_ENABLED=False)."""
        self._write_classified_json(tmp_path, ["NotInCodebook"])
        quotes = AIQuoteLoader(tmp_path, sample_tags).load_quotes()
        assert list(quotes.keys()) == ["doc"]
        (quote,) = quotes["doc"]
        (tag,) = quote.tags
        # Tag equality/hashing is by name only (see AGENTS.md) -- check
        # additional_tag explicitly, since a plain `==` wouldn't catch it.
        assert tag.name == "NotInCodebook"
        assert tag.additional_tag is True

    def test_undefined_tag_logs_to_process_log(
        self, tmp_path: Path, sample_tags, monkeypatch
    ):
        monkeypatch.setattr("src.io.loaders.UNDEFINED_TAGS_ENABLED", False)
        self._write_classified_json(tmp_path, ["NotInCodebook"])
        log_path = tmp_path / "process_log.xlsx"
        log = process_log.ProcessLog(log_path, step="run_main")
        with process_log.active(log):
            AIQuoteLoader(tmp_path, sample_tags).load_quotes()
        log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.UNDEFINED_TAGS)
        assert len(df) == 1
        assert df.iloc[0]["tag_name"] == "NotInCodebook"
        assert df.iloc[0]["reason"] == "unknown_name"
        assert df.iloc[0]["resolution"] == "dropped"

    def test_undefined_tags_kept_when_enabled(
        self, tmp_path: Path, sample_tags, monkeypatch
    ):
        monkeypatch.setattr("src.io.loaders.UNDEFINED_TAGS_ENABLED", True)
        self._write_classified_json(tmp_path, ["NotInCodebook"])

        quotes = AIQuoteLoader(tmp_path, sample_tags).load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag.name == "NotInCodebook"
        assert tag.additional_tag is True
        assert tag.color is not None

    def test_undefined_tag_color_is_stable_across_occurrences(
        self, tmp_path: Path, sample_tags, monkeypatch
    ):
        monkeypatch.setattr("src.io.loaders.UNDEFINED_TAGS_ENABLED", True)
        self._write_classified_json(tmp_path, ["Mystery", "Mystery"])

        quotes = AIQuoteLoader(tmp_path, sample_tags).load_quotes()

        first_tag = quotes["doc"][0].tags[0]
        second_tag = quotes["doc"][1].tags[0]
        assert first_tag.color == second_tag.color


class TestConsensusQuoteLoader:
    """Tests for loading consensus.json (see src.coding.consensus.build_consensus_records)."""

    @staticmethod
    def _write_consensus_json(file_path: Path, records: list[dict]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(records), encoding="utf-8")

    def test_loads_consensus_quote_with_resolved_fields(
        self, tmp_path: Path, sample_tags
    ):
        consensus_file = tmp_path / "consensus.json"
        self._write_consensus_json(
            consensus_file,
            [
                {
                    "phrase": "clear signs",
                    "tags": ["Monitoring"],
                    "source": {"source_name": "doc", "row": 0},
                    "start": 10,
                    "end": 21,
                    "source_iterations": 3,
                }
            ],
        )

        quotes = ConsensusQuoteLoader(consensus_file, sample_tags).load_quotes()

        quote = quotes["doc"][0]
        assert isinstance(quote, ConsensusQuote)
        assert quote.text == "clear signs"
        assert [tag.name for tag in quote.tags] == ["Monitoring"]
        assert quote.start == 10
        assert quote.end == 21
        assert quote.source_iterations == 3

    def test_skips_records_with_unresolved_position(self, tmp_path: Path, sample_tags):
        consensus_file = tmp_path / "consensus.json"
        self._write_consensus_json(
            consensus_file,
            [
                {
                    "phrase": "unresolved",
                    "tags": ["Monitoring"],
                    "source": {"source_name": "doc", "row": 0},
                    "start": None,
                    "end": None,
                    "source_iterations": 1,
                }
            ],
        )

        quotes = ConsensusQuoteLoader(consensus_file, sample_tags).load_quotes()

        assert quotes == {}

    def test_undefined_tags_dropped_when_disabled(
        self, tmp_path: Path, sample_tags, monkeypatch
    ):
        monkeypatch.setattr("src.io.loaders.UNDEFINED_TAGS_ENABLED", False)
        consensus_file = tmp_path / "consensus.json"
        self._write_consensus_json(
            consensus_file,
            [
                {
                    "phrase": "clear signs",
                    "tags": ["NotInCodebook"],
                    "source": {"source_name": "doc", "row": 0},
                    "start": 10,
                    "end": 21,
                }
            ],
        )

        quotes = ConsensusQuoteLoader(consensus_file, sample_tags).load_quotes()

        assert quotes == {}

    def test_undefined_tags_kept_by_default(self, tmp_path: Path, sample_tags):
        consensus_file = tmp_path / "consensus.json"
        self._write_consensus_json(
            consensus_file,
            [
                {
                    "phrase": "clear signs",
                    "tags": ["NotInCodebook"],
                    "source": {"source_name": "doc", "row": 0},
                    "start": 10,
                    "end": 21,
                }
            ],
        )

        quotes = ConsensusQuoteLoader(consensus_file, sample_tags).load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag.name == "NotInCodebook"
        assert tag.additional_tag is True


class TestAIQuoteLoaderConditionCompatibility:
    """Tests for condition-based tag resolution (see src.io.loaders._apply_condition())."""

    @staticmethod
    def _write_classified_json(directory: Path, doc_name: str, tag_name: str) -> None:
        (directory / f"{doc_name}_classified.json").write_text(
            json.dumps(
                [
                    {
                        "phrase": "some phrase",
                        "tags": [tag_name],
                        "source": {"sourcefile": f"{doc_name}.txt", "row": 0},
                    }
                ]
            ),
            encoding="utf-8",
        )

    def test_flags_mismatched_condition_as_undefined_but_keeps_real_metadata(
        self, tmp_path: Path
    ):
        restricted_tag = Tag(
            name="Restricted", color="#123456", description="desc", condition="advice"
        )
        self._write_classified_json(tmp_path, "doc", "Restricted")

        loader = AIQuoteLoader(
            tmp_path,
            [restricted_tag],
            condition_match=lambda name: ["no advice"],
        )
        quotes = loader.load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag.name == "Restricted"
        assert tag.additional_tag is True
        assert tag.color == "#123456"
        assert tag.description == "desc"

    def test_keeps_matching_condition_tag_unchanged(self, tmp_path: Path):
        restricted_tag = Tag(name="Restricted", condition="advice")
        self._write_classified_json(tmp_path, "doc", "Restricted")

        loader = AIQuoteLoader(
            tmp_path,
            [restricted_tag],
            condition_match=lambda name: ["advice"],
        )
        quotes = loader.load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag is restricted_tag
        assert tag.additional_tag is not True

    def test_drops_mismatched_tag_when_undefined_tags_disabled(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr("src.io.loaders.UNDEFINED_TAGS_ENABLED", False)
        restricted_tag = Tag(name="Restricted", condition="advice")
        self._write_classified_json(tmp_path, "doc", "Restricted")

        loader = AIQuoteLoader(
            tmp_path,
            [restricted_tag],
            condition_match=lambda name: ["no advice"],
        )

        assert loader.load_quotes() == {}

    def test_without_condition_lookup_configured_ignores_condition(self, tmp_path: Path):
        """No condition_match configured -> behaves exactly as before, with
        no condition filtering at all."""
        restricted_tag = Tag(name="Restricted", condition="advice")
        self._write_classified_json(tmp_path, "doc", "Restricted")

        loader = AIQuoteLoader(tmp_path, [restricted_tag])
        quotes = loader.load_quotes()

        assert quotes["doc"][0].tags[0] is restricted_tag


class TestConsensusQuoteLoaderConditionCompatibility:
    def test_flags_mismatched_condition_as_undefined_but_keeps_real_metadata(
        self, tmp_path: Path
    ):
        restricted_tag = Tag(name="Restricted", color="#abcdef", condition="advice")
        consensus_file = tmp_path / "consensus.json"
        consensus_file.write_text(
            json.dumps(
                [
                    {
                        "phrase": "clear signs",
                        "tags": ["Restricted"],
                        "source": {"source_name": "doc", "row": 0},
                        "start": 10,
                        "end": 21,
                    }
                ]
            ),
            encoding="utf-8",
        )

        loader = ConsensusQuoteLoader(
            consensus_file,
            [restricted_tag],
            condition_match=lambda name: ["no advice"],
        )
        quotes = loader.load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag.additional_tag is True
        assert tag.color == "#abcdef"


class TestAtlasQuoteLoaderConditionCompatibility:
    def test_flags_mismatched_condition_as_undefined_but_keeps_real_metadata(
        self, tmp_path: Path
    ):
        restricted_tag = Tag(name="Restricted", color="#112233", condition="advice")
        xlsx_path = tmp_path / "atlas.xlsx"
        pd.DataFrame(
            {
                "quote_text": ["some quote"],
                "quote_start": [0],
                "quote_end": [9],
                "Source Name": ["doc"],
                "Restricted": [1],
            }
        ).to_excel(xlsx_path, index=False)

        loader = AtlasQuoteLoader(
            xlsx_path,
            [restricted_tag],
            condition_match=lambda name: ["no advice"],
        )
        quotes = loader.load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag.additional_tag is True
        assert tag.color == "#112233"

    def test_keeps_matching_condition_tag_unchanged(self, tmp_path: Path):
        restricted_tag = Tag(name="Restricted", condition="advice")
        xlsx_path = tmp_path / "atlas.xlsx"
        pd.DataFrame(
            {
                "quote_text": ["some quote"],
                "quote_start": [0],
                "quote_end": [9],
                "Source Name": ["doc"],
                "Restricted": [1],
            }
        ).to_excel(xlsx_path, index=False)

        loader = AtlasQuoteLoader(
            xlsx_path,
            [restricted_tag],
            condition_match=lambda name: ["advice"],
        )
        quotes = loader.load_quotes()

        tag = quotes["doc"][0].tags[0]
        assert tag is restricted_tag
