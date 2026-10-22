"""Fixed-prior HMM sequence smoother for raw H/A/D evidence observations.

No labels are fitted here. The transition and emission assumptions are fixed in
config so the public example is used only to compare, not train, the experiment.
"""

import numpy as np
from hmmlearn.hmm import CategoricalHMM


N_STATES = 3  # Home, Away, dead/out of play
N_SYMBOLS = 10  # 3 evidence labels x 3 confidence bins, plus uninformative


def observation_symbols(costs, weak_margin=0.03, medium_margin=0.15):
    """Turn per-frame relative state costs into categorical evidence tokens."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != N_STATES or not np.isfinite(costs).all():
        raise ValueError("Expected finite frame-by-three H/A/D costs")
    if weak_margin < 0 or medium_margin < weak_margin:
        raise ValueError("Confidence margins must be ordered and nonnegative")
    sorted_costs = np.sort(costs, axis=1)
    margin = sorted_costs[:, 1] - sorted_costs[:, 0]
    state = np.argmin(costs, axis=1)
    confidence = np.where(margin >= medium_margin, 2,
                          np.where(margin >= weak_margin, 1, 0))
    symbols = state * 3 + confidence
    symbols[margin < weak_margin] = 9
    return symbols.astype(np.int64)


def make_model(config):
    """Create an HMM with hand-set persistence and emission assumptions."""
    team_self = float(config["team_self_transition"])
    team_to_opponent = float(config["team_to_opponent_transition"])
    team_to_dead = float(config["team_to_dead_transition"])
    dead_self = float(config["dead_self_transition"])
    dead_to_team = float(config["dead_to_team_transition"])
    if not np.isclose(team_self + team_to_opponent + team_to_dead, 1.0):
        raise ValueError("Team HMM transition probabilities must sum to one")
    if not np.isclose(dead_self + 2 * dead_to_team, 1.0):
        raise ValueError("Dead-state HMM transition probabilities must sum to one")

    transition = np.array([
        [team_self, team_to_opponent, team_to_dead],
        [team_to_opponent, team_self, team_to_dead],
        [dead_to_team, dead_to_team, dead_self],
    ], dtype=float)
    same_class = float(config["same_state_emission_probability"])
    unknown = float(config["unknown_emission_probability"])
    if not 0 < same_class < 1 or not 0 < unknown < 1:
        raise ValueError("HMM emission probabilities must be between zero and one")
    wrong_class = (1.0 - same_class) / 2.0
    same_confidence = np.asarray(config["same_confidence_mix"], dtype=float)
    wrong_confidence = np.asarray(config["wrong_confidence_mix"], dtype=float)
    if (same_confidence.shape != (3,) or wrong_confidence.shape != (3,) or
            not np.isclose(same_confidence.sum(), 1.0) or
            not np.isclose(wrong_confidence.sum(), 1.0)):
        raise ValueError("Each confidence mix must contain three probabilities summing to one")

    emissions = np.zeros((N_STATES, N_SYMBOLS), dtype=float)
    for hidden in range(N_STATES):
        for observed in range(N_STATES):
            class_probability = same_class if observed == hidden else wrong_class
            confidence_mix = same_confidence if observed == hidden else wrong_confidence
            emissions[hidden, observed * 3: observed * 3 + 3] = (
                (1.0 - unknown) * class_probability * confidence_mix)
        emissions[hidden, 9] = unknown

    model = CategoricalHMM(n_components=N_STATES, n_features=N_SYMBOLS,
                           algorithm="viterbi", init_params="", params="", n_iter=0)
    model.startprob_ = np.array([0.45, 0.45, 0.10], dtype=float)
    model.transmat_ = transition
    model.emissionprob_ = emissions
    return model


def smooth_costs(costs, blocks, config):
    """Decode each scored half independently; return H/A/D indices per frame."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != N_STATES or not np.isfinite(costs).all():
        raise ValueError("Expected finite frame-by-three H/A/D costs")
    symbols = observation_symbols(costs, config["weak_confidence_margin"],
                                  config["medium_confidence_margin"])
    model = make_model(config)
    result = np.full(len(costs), -1, dtype=np.int8)
    for _, start, last in blocks:
        if start < 0 or last >= len(costs) or start > last:
            raise ValueError("Invalid scored half bounds")
        observations = symbols[start:last + 1, None]
        result[start:last + 1] = model.predict(observations).astype(np.int8)
    return result
