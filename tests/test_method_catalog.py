from argparse import Namespace
import json

from survot_rank.cli import cmd_methods
from survot_rank.project import PROJECT_ROOT
from survot_rank.research.methods.catalog import (
    PRIMARY_METHOD,
    METHOD_ALIASES,
    METHOD_CATALOG,
    METHOD_CHOICES,
    METHOD_REGISTRY,
    METHOD_SPECS,
    catalog_errors,
)
from survot_rank.training.extended_args import build_base_parser
from survot_rank.training import model_factory


def test_catalog_is_the_complete_factory_source_of_truth():
    assert len(METHOD_SPECS) == len(METHOD_CATALOG) == len(METHOD_REGISTRY) == 21
    assert len(METHOD_ALIASES) == 31
    assert model_factory.METHOD_REGISTRY is METHOD_REGISTRY
    assert model_factory.METHOD_ALIASES is METHOD_ALIASES
    assert model_factory.list_methods() == [spec.key for spec in METHOD_SPECS]
    assert METHOD_CATALOG[PRIMARY_METHOD].status == "primary"
    assert catalog_errors(PROJECT_ROOT) == []


def test_human_method_index_mentions_every_canonical_key():
    method_index = (PROJECT_ROOT / "docs" / "METHODS.md").read_text(encoding="utf-8-sig")
    for spec in METHOD_SPECS:
        assert f"`{spec.key}`" in method_index


def test_argparse_choices_are_derived_from_registry_and_include_numeric_aliases():
    assert set(METHOD_CHOICES) == set(METHOD_REGISTRY) | set(METHOD_ALIASES)
    parser = build_base_parser()
    for alias, canonical in {
        "31": "ot_event_hazard_v2",
        "45": "otehv2_rankevent",
        "60": "v60_ot_event_rank",
        "70": "v70_patient_specific_prognostic_circuits",
    }.items():
        args = parser.parse_args(["--survot_method", alias])
        assert METHOD_ALIASES[args.survot_method] == canonical


def test_methods_command_supports_status_filter_and_json(capsys):
    cmd_methods(Namespace(status="primary", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert [item["key"] for item in payload] == [PRIMARY_METHOD]
    assert payload[0]["status"] == "primary"
