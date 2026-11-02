"""Strategyproof facility location on a line with blocked zones and predictions (CS 6001 course project).

Modules:
    environment: the road, its blocked zones and random layouts.
    agents: random agent profiles.
    mechanisms: the mechanism interface and exact optima.
    learning_augmented: snapping, MinMaxP*, the midpoint rule and randomised snapping (vectorised).
    fast_validator: exhaustive strategyproofness and truthfulness-in-expectation checker.
    experiment_utils: random instances and the adversarial search.
    candidate_lower_bound: the linear program and exact certificates for candidate sites.
    double_peaked: double-peaked costs, M1 and the window rules.
"""
