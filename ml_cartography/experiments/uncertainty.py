"""Compare cartography signals with agreement/uncertainty annotations."""

from __future__ import annotations

from typing import Dict, List

from scipy.stats import spearmanr


def spearman_with_human_agreement(rows: List[Dict], agreement_key: str = "human_agreement") -> Dict:
    conf = []
    var = []
    agree = []

    for r in rows:
        if agreement_key not in r or r[agreement_key] in ("", None):
            continue
        conf.append(float(r["confidence"]))
        var.append(float(r["variability"]))
        agree.append(float(r[agreement_key]))

    if len(agree) < 3:
        return {"n": len(agree), "rho_conf": None, "rho_var": None}

    rho_conf, _ = spearmanr(conf, agree)
    rho_var, _ = spearmanr(var, agree)
    return {"n": len(agree), "rho_conf": float(rho_conf), "rho_var": float(rho_var)}

