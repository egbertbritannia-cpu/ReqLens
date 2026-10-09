# SERF code-flow review

Static review: the development notebook uses an ELECTRA selector and RoBERTa probability interpolation; the training notebook routes to a separately trained ELECTRA–RoBERTa fusion model. These implementations require separate provenance before comparison.
