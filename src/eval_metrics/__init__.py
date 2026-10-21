"""Rank-based metrics and per-query rankings for FashionMV retrieval runs.

ranking.py turns query and gallery embeddings into per-query ranks and top-K lists (saved once per
model and dataset); metrics.py computes every metric from those ranks on CPU; report.py writes the
tables. upstream_dump.py produces the same files with the authors' evaluate.py protocol.
"""
