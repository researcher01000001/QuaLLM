from pathlib import Path

import pandas as pd

import run
from src.config.coding import CodingConfiguration
from src.config.general import SUPERCONSENSUS_NAME
from src.core import process_log
from src.core.models import (
    AtlasQuote,
    CodedRow,
    CodingScheme,
    ConsensusQuote,
    DisplayColumn,
    Document,
    Row,
    Tag,
)

_EMPTY_DISPLAY_COLUMN = DisplayColumn(source_column="", label="", values={})


def test_build_render_codings_builds_one_row_per_consensus_source():
    tag = Tag("A", "#111111")
    html_consensus_quotes = {
        "superconsensus": {
            "doc-1": [ConsensusQuote(text="Te", tags=[tag], start=0, end=2)]
        },
    }

    codings = run.build_render_codings(
        atlas_quotes={"doc-1": []},
        html_consensus_quotes=html_consensus_quotes,
    )

    assert codings["AI consensus (superconsensus)"] == html_consensus_quotes["superconsensus"]


def test_build_render_codings_supports_multiple_consensus_sources():
    tag = Tag("A", "#111111")
    html_consensus_quotes = {
        "config-a": {"doc-1": [ConsensusQuote(text="Te", tags=[tag], start=0, end=2)]},
        "config-b": {"doc-1": [ConsensusQuote(text="st", tags=[tag], start=2, end=4)]},
    }

    codings = run.build_render_codings(
        atlas_quotes={"doc-1": []},
        html_consensus_quotes=html_consensus_quotes,
    )

    assert set(codings) == {
        "Human (ATLAS)",
        "AI consensus (config-a)",
        "AI consensus (config-b)",
    }


def test_apply_coder_comparison_scores_converts_paragraphs_to_coded():
    tag = Tag("A", "#111111")
    document = Document(
        source_name="doc-1",
        rows=[
            Row(index=0, text="Hello world", start_position=0, end_position=11)
        ],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    documents = {"doc-1": document}

    human_quote = AtlasQuote(text="Hello", tags=[tag], start=0, end=5)
    ai_quote = ConsensusQuote(text="world", tags=[tag], start=6, end=11)

    run.apply_coder_comparison_scores(
        documents,
        atlas_quotes={"doc-1": [human_quote]},
        html_consensus_quotes={"superconsensus": {"doc-1": [ai_quote]}},
    )

    paragraph = documents["doc-1"].rows[0]
    assert isinstance(paragraph, CodedRow)
    assert paragraph.quotes_1 == [human_quote]
    assert paragraph.quotes_2 == [ai_quote]


def test_apply_coder_comparison_scores_noop_without_consensus_sources():
    document = Document(
        source_name="doc-1",
        rows=[
            Row(index=0, text="Hello world", start_position=0, end_position=11)
        ],
        display_column=_EMPTY_DISPLAY_COLUMN,
    )
    documents = {"doc-1": document}

    run.apply_coder_comparison_scores(documents, atlas_quotes={}, html_consensus_quotes={})

    assert not isinstance(documents["doc-1"].rows[0], CodedRow)


class TestExportConsensusJsonDoubleCodingConflicts:
    """Double-coding conflicts must only be flagged in the genuinely final,
    decided consensus (a configuration's own consensus.json, or the
    cross-configuration superconsensus) -- never in intermediate progression
    snapshots, which export_consensus_json() never even receives."""

    def _document(self) -> Document:
        return Document(
            source_name="doc-1",
            rows=[
                Row(index=0, text="Hello world", start_position=0, end_position=11)
            ],
            display_column=_EMPTY_DISPLAY_COLUMN,
        )

    def test_logs_conflict_in_single_configuration_final_consensus(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(run, "AI_CODING_FILES", tmp_path)
        monkeypatch.setattr(run, "PREFIX", "prefix")
        monkeypatch.setattr(run, "CONSENSUS_FILENAME", "consensus.json")

        tag_a = Tag("A", can_co_occur=False)
        tag_b = Tag("B", can_co_occur=False)
        config = CodingConfiguration(short_name="cfg", iterations=1)
        document = self._document()
        quote = ConsensusQuote(text="Hello", tags=[tag_a, tag_b], start=0, end=5)

        log_path = tmp_path / "process_log.xlsx"
        log = process_log.ProcessLog(
            log_path, step="run_main", owned_steps={"run_main", "run_main/cfg"}
        )
        with process_log.active(log):
            run.export_consensus_json(
                documents={"doc-1": document},
                ai_quote_runs={("cfg", 0): {"doc-1": [quote]}},
                consensus_quotes={},
                superconsensus_quotes={},
                coding_configurations=[config],
            )
        log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.DOUBLE_CODING_CONFLICTS)
        assert len(df) == 1
        assert df.iloc[0]["source_name"] == "doc-1"
        assert df.iloc[0]["step"] == "run_main/cfg"
        assert {df.iloc[0]["tag_a"], df.iloc[0]["tag_b"]} == {"A", "B"}

    def test_logs_conflict_only_in_superconsensus_not_duplicated_per_configuration(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(run, "AI_CODING_FILES", tmp_path)
        monkeypatch.setattr(run, "PREFIX", "prefix")
        monkeypatch.setattr(run, "CONSENSUS_FILENAME", "consensus.json")

        clean_tag = Tag("Clean", can_co_occur=True)
        tag_a = Tag("A", can_co_occur=False)
        tag_b = Tag("B", can_co_occur=False)
        config_a = CodingConfiguration(short_name="cfg-a", iterations=2)
        config_b = CodingConfiguration(short_name="cfg-b", iterations=2)
        document = self._document()

        # Each configuration's own final consensus is conflict-free...
        consensus_quotes = {
            "cfg-a": {"doc-1": [ConsensusQuote(text="Hello", tags=[clean_tag], start=0, end=5)]},
            "cfg-b": {"doc-1": [ConsensusQuote(text="Hello", tags=[clean_tag], start=0, end=5)]},
        }
        # ...but the cross-configuration superconsensus vote produces a conflict.
        superconsensus_quotes = {
            "doc-1": [ConsensusQuote(text="Hello", tags=[tag_a, tag_b], start=0, end=5)]
        }

        log_path = tmp_path / "process_log.xlsx"
        log = process_log.ProcessLog(
            log_path, step="run_main", owned_steps={"run_main", "run_main/superconsensus"}
        )
        with process_log.active(log):
            run.export_consensus_json(
                documents={"doc-1": document},
                ai_quote_runs={},
                consensus_quotes=consensus_quotes,
                superconsensus_quotes=superconsensus_quotes,
                coding_configurations=[config_a, config_b],
            )
        log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.DOUBLE_CODING_CONFLICTS)
        assert len(df) == 1
        assert df.iloc[0]["step"] == "run_main/superconsensus"
        assert "superconsensus" in df.iloc[0]["message"]


class TestRunMainOwnedSteps:
    """_run_main_owned_steps() decides which run_main sub-steps this
    invocation is allowed to replace on flush -- a configuration dropped
    from CODING_CONFIGURATIONS on a later run must not have its old rows
    wiped just because run_main reran as a whole."""

    def test_includes_run_main_and_superconsensus_always(self):
        owned = run._run_main_owned_steps([], [])
        assert owned == {"run_main", "run_main/superconsensus"}

    def test_includes_one_entry_per_configuration(self):
        config_a = CodingConfiguration(short_name="cfg-a")
        config_b = CodingConfiguration(short_name="cfg-b")
        owned = run._run_main_owned_steps([config_a, config_b], [])
        assert owned == {
            "run_main",
            "run_main/superconsensus",
            "run_main/cfg-a",
            "run_main/cfg-b",
        }

    def test_includes_html_consensus_configurations_defensively(self):
        owned = run._run_main_owned_steps([], ["some-other-config"])
        assert "run_main/some-other-config" in owned

    def test_dropping_a_configuration_preserves_its_previously_logged_rows(self, tmp_path):
        log_path = tmp_path / "process_log.xlsx"

        config_a = CodingConfiguration(short_name="cfg-a")
        first_run_owned_steps = run._run_main_owned_steps([config_a], [])
        first_log = process_log.ProcessLog(
            log_path, step="run_main", owned_steps=first_run_owned_steps
        )
        first_log.log_general_warning(
            severity="Warning", message="from cfg-a", step="run_main/cfg-a"
        )
        first_log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.GENERAL_WARNINGS)
        assert len(df) == 1
        assert df.iloc[0]["step"] == "run_main/cfg-a"

        # A later run no longer configures cfg-a at all.
        second_run_owned_steps = run._run_main_owned_steps([], [])
        second_log = process_log.ProcessLog(
            log_path, step="run_main", owned_steps=second_run_owned_steps
        )
        second_log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.GENERAL_WARNINGS)
        assert len(df) == 1
        assert df.iloc[0]["step"] == "run_main/cfg-a"


class TestSliceToScope:
    """_slice_to_scope() narrows every per-document mapping down to one
    report scope's document names (see src.dataset.run_scope.report_scopes()),
    without KeyErrors for documents missing from any particular mapping."""

    def _document(self, name: str) -> Document:
        return Document(
            source_name=name,
            rows=[Row(index=0, text="Hello", start_position=0, end_position=5)],
            display_column=_EMPTY_DISPLAY_COLUMN,
        )

    def test_narrows_documents_to_scope(self):
        documents = {"doc-1": self._document("doc-1"), "doc-2": self._document("doc-2")}

        scoped = run._slice_to_scope(
            documents=documents,
            atlas_quotes={},
            ai_quote_runs={},
            consensus_progression_quotes={},
            superconsensus_quotes={},
            html_consensus_quotes={},
            scope_doc_names=["doc-1"],
        )

        assert set(scoped.documents) == {"doc-1"}

    def test_ignores_scope_documents_missing_from_documents_dict(self):
        documents = {"doc-1": self._document("doc-1")}

        scoped = run._slice_to_scope(
            documents=documents,
            atlas_quotes={},
            ai_quote_runs={},
            consensus_progression_quotes={},
            superconsensus_quotes={},
            html_consensus_quotes={},
            scope_doc_names=["doc-1", "doc-missing"],
        )

        assert set(scoped.documents) == {"doc-1"}

    def test_missing_quotes_default_to_empty_list_not_keyerror(self):
        scoped = run._slice_to_scope(
            documents={},
            atlas_quotes={"doc-1": [ConsensusQuote(text="x", tags=[], start=0, end=1)]},
            ai_quote_runs={("cfg", 0): {}},
            consensus_progression_quotes={0: {"cfg": {}}},
            superconsensus_quotes={},
            html_consensus_quotes={"superconsensus": {}},
            scope_doc_names=["doc-1", "doc-2"],
        )

        assert scoped.atlas_quotes["doc-2"] == []
        assert scoped.ai_quote_runs[("cfg", 0)]["doc-2"] == []
        assert scoped.consensus_progression_quotes[0]["cfg"]["doc-2"] == []
        assert scoped.superconsensus_quotes["doc-1"] == []
        assert scoped.html_consensus_quotes["superconsensus"]["doc-1"] == []

    def test_only_requested_scope_names_appear_in_sliced_mappings(self):
        scoped = run._slice_to_scope(
            documents={},
            atlas_quotes={"doc-1": [], "doc-2": [], "doc-3": []},
            ai_quote_runs={},
            consensus_progression_quotes={},
            superconsensus_quotes={},
            html_consensus_quotes={},
            scope_doc_names=["doc-1", "doc-2"],
        )

        assert set(scoped.atlas_quotes) == {"doc-1", "doc-2"}


class TestComputeTagAnalysisAndConfusionRows:
    """compute_tag_analysis_and_confusion_rows() must not treat
    scoped.superconsensus_quotes as "there is a real superconsensus" just
    because the dict itself is non-empty -- _slice_to_scope() always
    produces one entry per scoped document (see
    TestSliceToScope.test_missing_quotes_default_to_empty_list_not_keyerror
    above), so a source superconsensus that's genuinely empty (zero coding
    configurations -- see run.build_superconsensus_quote_set(); one
    configuration's superconsensus is that configuration's own consensus,
    not empty) still ends up as a non-empty dict of empty lists after
    scoping."""

    def _document(self, name: str, text: str) -> Document:
        return Document(
            source_name=name,
            rows=[Row(index=0, text=text, start_position=0, end_position=len(text))],
            display_column=DisplayColumn(
                source_column="position", label="Paragraph", values={0: "1"}
            ),
        )

    def test_no_real_superconsensus_produces_no_superconsensus_rows(self):
        tag = Tag(name="Monitoring")
        doc = self._document("1_advice_cycle_1", "abcde")
        scoped = run._ScopedPipelineData(
            documents={doc.source_name: doc},
            atlas_quotes={
                doc.source_name: [AtlasQuote(text="abcde", tags=[tag], start=0, end=5)]
            },
            ai_quote_runs={},
            consensus_progression_quotes={
                0: {
                    "cfgA": {
                        doc.source_name: [
                            ConsensusQuote(text="abcde", tags=[tag], start=0, end=5)
                        ]
                    }
                }
            },
            # This is exactly what _slice_to_scope() produces when the
            # source superconsensus_quotes was {} (only one coding
            # configuration configured) -- one empty-list entry per scoped
            # document, not an empty dict.
            superconsensus_quotes={doc.source_name: []},
            html_consensus_quotes={},
        )

        tag_analysis_rows, tag_confusion_rows = run.compute_tag_analysis_and_confusion_rows(
            scoped, [tag], final_iterations_by_configuration={"cfgA": 0}, drop_undefined_tags=False
        )

        configurations = {row["configuration"] for row in tag_analysis_rows}
        assert configurations == {"cfgA"}
        assert SUPERCONSENSUS_NAME not in configurations
        assert tag_confusion_rows == []


class TestRunAiCoding:
    """run_ai_coding() logs one process-log entry per configuration and
    writes one prompt-snapshot file per document condition (as reported by
    document_content.known_conditions()), only when RUN_AI_CODING is active
    -- see docs/pipeline.md's AI-coding prompt snapshot section."""

    def test_logs_once_per_configuration_and_writes_one_snapshot_call(
        self, tmp_path, monkeypatch
    ):
        config_a = CodingConfiguration(short_name="cfg-a", iterations=1)
        config_b = CodingConfiguration(short_name="cfg-b", iterations=1)

        monkeypatch.setattr(run, "RUN_AI_CODING", True)
        # Isolate this test from the real (possibly True) COST_ESTIMATION_ENABLED
        # setting, which would otherwise call cost_estimation.estimate_run_cost()
        # and block on input() for a confirmation prompt.
        monkeypatch.setattr(run, "COST_ESTIMATION_ENABLED", False)
        monkeypatch.setattr(run, "CODING_CONFIGURATIONS", [config_a, config_b])
        monkeypatch.setattr(run, "process_documents", lambda *a, **k: None)
        monkeypatch.setattr(
            run,
            "load_prompt_context",
            lambda *a, **k: CodingScheme(coding_scheme=[], general_rules=[], medical_knowledge={}),
        )
        # The dataset's full condition vocabulary -- deliberately includes a
        # condition ("archived") not present among the "current run" files
        # below, to prove it's still generated (just later markable as
        # unused by write_prompt_snapshots(), which this test doesn't
        # exercise directly -- see TestWritePromptSnapshotsActiveConditions).
        monkeypatch.setattr(run, "known_conditions", lambda: ["advice", "no advice", "archived"])
        monkeypatch.setattr(
            run.run_scope,
            "find_source_documents",
            lambda *a, **k: [Path("12_advice_cycle_1.txt"), Path("7_no advice_cycle_2.txt")],
        )

        captured_entries = []
        captured_active_conditions = []

        def fake_write(entries, active_conditions=None, *a, **k):
            captured_entries.append(entries)
            captured_active_conditions.append(active_conditions)
            return [tmp_path / "prompt_full.txt"]

        monkeypatch.setattr(run, "write_prompt_snapshots", fake_write)

        log_path = tmp_path / "process_log.xlsx"
        log = process_log.ProcessLog(log_path, step="run_main", owned_steps={"run_main"})
        with process_log.active(log):
            run.run_ai_coding([Tag("A")])
        log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.GENERAL_WARNINGS)
        assert len(df) == 2
        assert set(df["severity"]) == {"Info"}
        assert len(captured_entries) == 1
        entries = captured_entries[0]
        # One entry per (condition, configuration) pair, covering every
        # dataset condition -- not just the ones in the current run's files.
        assert len(entries) == 3 * 2
        assert {entry[0] for entry in entries} == {"advice", "no advice", "archived"}
        assert {entry[1] for entry in entries} == {config_a, config_b}
        # Only the conditions actually present in the current run's files.
        assert captured_active_conditions == [{"advice", "no advice"}]

    def test_disabled_logs_nothing_and_never_writes_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run, "RUN_AI_CODING", False)
        write_calls = []
        monkeypatch.setattr(
            run, "write_prompt_snapshots", lambda *a, **k: write_calls.append(a)
        )

        log_path = tmp_path / "process_log.xlsx"
        log = process_log.ProcessLog(log_path, step="run_main", owned_steps={"run_main"})
        with process_log.active(log):
            run.run_ai_coding([])
        log.flush()

        df = pd.read_excel(log_path, sheet_name=process_log.GENERAL_WARNINGS)
        assert len(df) == 0
        assert write_calls == []
