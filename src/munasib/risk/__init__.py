"""Consumer-risk assessment: decision criteria, risk factors and fuzzy scoring.

The package has two layers:

* ``indicators`` / ``factors`` / ``compute`` - the P15 per-product criteria and
  rule-based risk factors, computed by ``scripts/compute_risk_indicators.py``
  from stored tables only and persisted in ``product_risk_indicators``;
* the fuzzy-logic engine that turns those 0-1 criteria into a 0-100 consumer
  risk score (``configs/fuzzy_rules.yaml``), consumed by the decision stage.

Formulas and thresholds are documented in ``docs/risk_indicators.md``.
"""
