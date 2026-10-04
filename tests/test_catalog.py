"""Catalog-driven smoke test for gassolar."""

from pathlib import Path

import pytest
from gpkit.tests.test_catalog import (
    catalog_ids,
    catalog_params,
    load_catalog,
    run_catalog_snapshots,
    run_catalog_test,
    run_catalog_to_ir,
    run_catalog_toml_roundtrip,
)

try:
    _CATALOG = load_catalog(Path(__file__))
except FileNotFoundError:
    _CATALOG = []


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_model(model_entry):
    run_catalog_test(model_entry)


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_snapshots(model_entry):
    """Regenerate each catalog entry's snapshots; drift shows as a git diff."""
    run_catalog_snapshots(model_entry, __file__)


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_to_ir(model_entry):
    """Each catalog entry exports a complete, self-consistent IR document."""
    run_catalog_to_ir(model_entry)


# Open gpkit-core defects in the TOML printer, not in these models. Every
# gassolar model carries a vector (cave) whose parent is owned by another
# model section, so all three trip the same one.
_TOML_ROUNDTRIP_GAPS = dict.fromkeys(
    catalog_ids(_CATALOG),
    "gpkit-core#295: to_toml raises on a vector element whose parent is elsewhere",
)


@pytest.mark.parametrize(
    "model_entry", catalog_params(_CATALOG, xfail=_TOML_ROUNDTRIP_GAPS)
)
def test_catalog_toml_roundtrip(model_entry):
    """Each catalog entry survives to_toml -> load_toml and solves the same."""
    run_catalog_toml_roundtrip(model_entry)
