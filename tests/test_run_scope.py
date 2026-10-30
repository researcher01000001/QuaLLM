"""Unit tests for src.dataset.run_scope"""

from pathlib import Path

import pytest

from src.dataset import run_scope


def _make_txt_files(
    tmp_path: Path,
    training: list[str],
    validation: list[str],
    full: list[str] | None = None,
    full_without_validation: list[str] | None = None,
    test: list[str] | None = None,
) -> Path:
    txt_files = tmp_path / "atlas_import"
    folders = {
        "training": training,
        "validation": validation,
        "full": full or [],
        "full_without_validation": full_without_validation or [],
        "test": test or [],
    }
    for folder_name, names in folders.items():
        folder = txt_files / folder_name
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.txt").write_text("text", encoding="utf-8")
    return txt_files


def _patch(
    monkeypatch,
    tmp_path: Path,
    training: list[str],
    validation: list[str],
    full: list[str] | None = None,
    full_without_validation: list[str] | None = None,
    test: list[str] | None = None,
) -> None:
    txt_files = _make_txt_files(tmp_path, training, validation, full, full_without_validation, test)
    monkeypatch.setattr(run_scope, "TXT_FILES", txt_files)


class TestFolders:
    def test_training_folder_is_txt_files_slash_training(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.training_folder() == run_scope.TXT_FILES / "training"

    def test_validation_folder_is_txt_files_slash_validation(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.validation_folder() == run_scope.TXT_FILES / "validation"

    def test_full_folder_is_txt_files_slash_full(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.full_folder() == run_scope.TXT_FILES / "full"

    def test_full_without_validation_folder_is_its_own_subfolder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert (
            run_scope.full_without_validation_folder()
            == run_scope.TXT_FILES / "full_without_validation"
        )

    def test_test_folder_is_txt_files_slash_test(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.test_folder() == run_scope.TXT_FILES / "test"


class TestDocumentSets:
    def test_training_documents_lists_txt_files_in_training_folder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1", "train-2"], ["val-1"])
        assert set(run_scope.training_documents()) == {"train-1", "train-2"}

    def test_validation_documents_lists_txt_files_in_validation_folder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1"], ["val-1", "val-2"])
        assert set(run_scope.validation_documents()) == {"val-1", "val-2"}

    def test_full_documents_lists_txt_files_in_full_folder(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch,
            tmp_path,
            ["train-1"],
            ["val-1"],
            full=["doc-1", "doc-2", "doc-3"],
        )
        assert set(run_scope.full_documents()) == {"doc-1", "doc-2", "doc-3"}

    def test_full_without_validation_documents_lists_its_own_folder(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch,
            tmp_path,
            ["train-1"],
            ["val-1"],
            full=["doc-1", "doc-2", "doc-3"],
            full_without_validation=["doc-1", "doc-2"],
        )
        assert set(run_scope.full_without_validation_documents()) == {"doc-1", "doc-2"}

    def test_test_documents_lists_txt_files_in_test_folder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1"], ["val-1"], test=["test-1", "test-2"])
        assert set(run_scope.test_documents()) == {"test-1", "test-2"}

    def test_missing_folder_returns_empty_list(self, monkeypatch, tmp_path):
        # Neither atlas_preperation.py run yet for this type.
        txt_files = tmp_path / "atlas_import"
        (txt_files / "training").mkdir(parents=True)
        monkeypatch.setattr(run_scope, "TXT_FILES", txt_files)
        assert run_scope.validation_documents() == []
        assert run_scope.full_documents() == []


class TestProcessingDirs:
    def test_single_folder_run_types_point_at_their_own_folder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.processing_dirs("training") == [run_scope.training_folder()]
        assert run_scope.processing_dirs("validation") == [run_scope.validation_folder()]
        assert run_scope.processing_dirs("test") == [run_scope.test_folder()]
        assert run_scope.processing_dirs("full_without_validation") == [
            run_scope.full_without_validation_folder()
        ]

    def test_full_covers_training_validation_and_full_but_not_trimmed_copies(
        self, monkeypatch, tmp_path
    ):
        _patch(monkeypatch, tmp_path, [], [])
        assert run_scope.processing_dirs("full") == [
            run_scope.training_folder(),
            run_scope.validation_folder(),
            run_scope.full_folder(),
        ]

    def test_unknown_run_type_raises(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        with pytest.raises(ValueError):
            run_scope.processing_dirs("not-a-real-run-type")


class TestFindSourceDocuments:
    def test_full_returns_each_document_name_once(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch,
            tmp_path,
            training=["doc_a"],
            validation=["validation_x"],
            full=["doc_a", "doc_b"],
            full_without_validation=["doc_a", "doc_b"],
        )
        paths = run_scope.find_source_documents("full")
        assert sorted(p.stem for p in paths) == ["doc_a", "doc_b", "validation_x"]
        assert len({p.stem for p in paths}) == len(paths)

    def test_full_prefers_the_training_copy_of_a_shared_document(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, training=["doc_a"], validation=[], full=["doc_a"])
        (path,) = run_scope.find_source_documents("full")
        assert path.parent == run_scope.training_folder()

    def test_full_never_reads_full_without_validation(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch, tmp_path, training=[], validation=[], full=["doc_a"],
            full_without_validation=["doc_a", "only_trimmed"],
        )
        paths = run_scope.find_source_documents("full")
        assert [p.stem for p in paths] == ["doc_a"]
        assert paths[0].parent == run_scope.full_folder()

    def test_single_folder_run_type_lists_its_folder(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, training=["t1", "t2"], validation=["v1"])
        assert [p.stem for p in run_scope.find_source_documents("training")] == ["t1", "t2"]


class TestFindCanonicalDocuments:
    def test_whole_document_in_full_wins_over_trimmed_copies(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch, tmp_path, training=["doc_a"], validation=["validation_x"],
            full=["doc_a"], full_without_validation=["doc_a", "doc_c"], test=["doc_t"],
        )
        by_stem = {p.stem: p for p in run_scope.find_canonical_documents()}
        assert sorted(by_stem) == ["doc_a", "doc_c", "doc_t", "validation_x"]
        assert by_stem["doc_a"].parent == run_scope.full_folder()


class TestReportScopes:
    def test_training_reports_only_training(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1", "train-2"], ["val-1"])
        scopes = run_scope.report_scopes("training")
        assert set(scopes) == {"training"}
        assert set(scopes["training"]) == {"train-1", "train-2"}

    def test_validation_reports_only_validation(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1"], ["val-1", "val-2"])
        scopes = run_scope.report_scopes("validation")
        assert set(scopes) == {"validation"}
        assert set(scopes["validation"]) == {"val-1", "val-2"}

    def test_full_without_validation_reports_training_only_despite_bigger_processing_pool(
        self, monkeypatch, tmp_path
    ):
        """full_without_validation now *processes* a bigger pool (training +
        untagged documents, via full_without_validation_folder()) than it
        *reports* on (training only) -- see report_scopes()'s docstring."""
        _patch(
            monkeypatch,
            tmp_path,
            ["train-1", "train-2"],
            ["val-1"],
            full_without_validation=["train-1", "train-2", "untagged-1", "untagged-2"],
        )
        scopes = run_scope.report_scopes("full_without_validation")
        assert set(scopes) == {"full_without_validation"}
        assert set(scopes["full_without_validation"]) == {"train-1", "train-2"}

    def test_full_produces_three_scopes(self, monkeypatch, tmp_path):
        _patch(
            monkeypatch,
            tmp_path,
            ["train-1", "train-2"],
            ["val-1", "val-2"],
            full=["train-1", "train-2", "val-1", "val-2", "untagged-1"],
        )
        scopes = run_scope.report_scopes("full")
        assert set(scopes) == {"training", "validation", "combined"}
        assert set(scopes["training"]) == {"train-1", "train-2"}
        assert set(scopes["validation"]) == {"val-1", "val-2"}
        # "combined" is training union validation only, NOT every document
        # in full_folder() -- "combination of both" means training+validation.
        assert set(scopes["combined"]) == {"train-1", "train-2", "val-1", "val-2"}

    def test_test_reports_only_test(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, ["train-1"], ["val-1"], test=["test-1", "test-2"])
        scopes = run_scope.report_scopes("test")
        assert set(scopes) == {"test"}
        assert set(scopes["test"]) == {"test-1", "test-2"}

    def test_unknown_run_type_raises(self, monkeypatch, tmp_path):
        _patch(monkeypatch, tmp_path, [], [])
        with pytest.raises(ValueError):
            run_scope.report_scopes("not-a-real-run-type")
