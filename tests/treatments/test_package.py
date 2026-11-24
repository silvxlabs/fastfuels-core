"""Tests for the :mod:`fastfuels_core.treatments` package itself.

The treatments are tested in their own modules. These pin what the
package promises its callers: every treatment imports from the package
root, and importing the package never loads numba.
"""

from __future__ import annotations

import subprocess
import sys


def test_public_names_import_from_the_package():
    # The names FastFuels-API-v2's standgen imports, the protocol that
    # trees.py imports, the tree-density thin and the clump-pattern thin.
    from fastfuels_core import treatments
    from fastfuels_core.treatments import (
        DirectionalThinToDiameterLimit,
        DirectionalThinToStandBasalArea,
        DirectionalThinToTreeDensity,
        ProportionalThinToBasalArea,
        ThinningDirection,
        ThinToClumpPattern,
        ThinToCrownSpacing,
        TreatmentProtocol,
    )

    imported = [
        DirectionalThinToDiameterLimit,
        DirectionalThinToStandBasalArea,
        DirectionalThinToTreeDensity,
        ProportionalThinToBasalArea,
        ThinningDirection,
        ThinToClumpPattern,
        ThinToCrownSpacing,
        TreatmentProtocol,
    ]
    assert {cls.__name__ for cls in imported} <= set(treatments.__all__)


def test_importing_the_package_does_not_import_numba():
    # A treatment that compiles with numba must import it lazily, so that
    # importing any treatment never pays for compilation. This runs in a
    # fresh interpreter because other tests may import numba into this one.
    code = "import sys, fastfuels_core.treatments; assert 'numba' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
