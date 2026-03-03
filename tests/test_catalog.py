"""Catalog-driven smoke test for gassolar."""

from pathlib import Path

import pytest
from gpkit.tests.test_catalog import catalog_ids, load_catalog, run_catalog_test

_CATALOG = load_catalog(Path(__file__))


@pytest.mark.parametrize("model_entry", _CATALOG, ids=catalog_ids(_CATALOG))
def test_catalog_model(model_entry):
    run_catalog_test(model_entry)
