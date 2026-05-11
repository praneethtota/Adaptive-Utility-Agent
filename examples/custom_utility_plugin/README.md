# AUA Custom Utility Scorer Plugin

Demonstrates how to replace the built-in U = w_e·E + w_c·C + w_k·K scorer
with a custom risk-weighted scorer.

## The plugin

```python
# plugins/custom_utility.py
class RiskWeightedUtilityScorer:
    """Weights low-confidence responses more heavily for high-risk domains."""

    def __init__(self, risk_weight: float = 0.7):
        self.risk_weight = risk_weight

    def score(self, response, field, prior_u, confidence, metadata):
        base_u = prior_u * 0.5 + confidence * 0.5
        if field in ("surgery", "medicine", "law"):
            return base_u * (1 - self.risk_weight * (1 - confidence))
        return base_u
```

## Registration in aua_config.yaml

```yaml
utility_scorer:
  import_path: plugins.custom_utility:RiskWeightedUtilityScorer
  config:
    risk_weight: 0.7
```

## Testing

```bash
aua extensions test \
  --kind utility_scorer \
  --import-path plugins.custom_utility:RiskWeightedUtilityScorer
```

Expected:
```
✓ Plugin loaded successfully
  Type:     RiskWeightedUtilityScorer
  Protocol: utility_scorer — contract satisfied ✓
```
