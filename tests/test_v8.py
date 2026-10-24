"""Behavior tests for ball-based stoppage and restart hypotheses."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from solution.v8_stoppages import build_evidence, decode_stops

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'solution/config_v8.json').read_text())
QUALITY = json.loads((ROOT/'solution/config_v7.json').read_text())


def sample(n=250):
    positions = np.array([[10+4*p, 10+3*p] for p in range(11)] +
                         [[60+3*p, 10+3*p] for p in range(11)], np.float32)
    positions[0] = [30.4, 30.0]  # nearby free-kick taker is allowed
    return SimpleNamespace(player_xy_m=np.tile(positions, (n, 1, 1)),
        home_slots=np.r_[np.ones(11, bool), np.zeros(11, bool)],
        ball_xy_m=np.tile(np.array([30., 30.], np.float32), (n, 1)),
        period=np.ones(n, int), fps=25., pitch=(105., 68.))


class Match(SimpleNamespace):
    def __len__(self):
        return len(self.period)


def evidence(n=250):
    return {'ball_reliable': np.ones(n), 'outside_margin_m': np.full(n, -20.),
            'stability_covered': np.ones(n), 'pressure_fraction': np.zeros(n),
            'robust_ball_span_m': np.full(n, 3.), 'quiet_fraction': np.zeros(n),
            'restart_candidate': np.zeros(n)}


class BallEvidenceTests(unittest.TestCase):
    def test_jitter_and_nearby_taker_do_not_disable_stability(self):
        m = Match(**vars(sample()))
        rng = np.random.default_rng(2027)
        m.ball_xy_m += rng.uniform(-.08, .08, m.ball_xy_m.shape).astype(np.float32)
        f = build_evidence(m, CONFIG, QUALITY)
        mask, _, _ = decode_stops(f, m.period, m.fps, CONFIG)
        self.assertTrue(mask[80:200].all())
        self.assertLess(f['nearest_ball_m'][100], 1.)

    def test_missing_ball_alone_never_enters_dead(self):
        m = Match(**vars(sample()))
        m.ball_xy_m[:] = np.nan
        f = build_evidence(m, CONFIG, QUALITY)
        mask, _, _ = decode_stops(f, m.period, m.fps, CONFIG, persistent=True)
        self.assertFalse(mask.any())

    def test_period_reset_prevents_stale_stability_and_dead_hold(self):
        m = Match(**vars(sample(150)))
        m.period[75:] = 2
        m.ball_xy_m[75:] = np.nan
        f = build_evidence(m, CONFIG, QUALITY)
        mask, _, _ = decode_stops(f, m.period, m.fps, CONFIG, persistent=True)
        self.assertTrue(mask[65])
        self.assertFalse(mask[75:].any())

    def test_single_boundary_jump_is_rejected(self):
        m = Match(**vars(sample()))
        m.ball_xy_m[100] = [-3, 30]
        f = build_evidence(m, CONFIG, QUALITY)
        mask, _, _ = decode_stops(f, m.period, m.fps, CONFIG, stationary=False)
        self.assertFalse(mask.any())

    def test_carrying_slowly_near_a_player_is_not_restart(self):
        m = Match(**vars(sample()))
        movement = np.arange(len(m), dtype=np.float32)*1.5/m.fps
        m.ball_xy_m[:, 0] += movement
        m.player_xy_m[:, 0, 0] += movement
        f = build_evidence(m, CONFIG, QUALITY)
        self.assertFalse(f['restart_candidate'].any())

    def test_fast_inside_release_is_restart_evidence(self):
        m = Match(**vars(sample(150)))
        m.ball_xy_m[40:, 0] += np.arange(110, dtype=np.float32)*10/m.fps
        f = build_evidence(m, CONFIG, QUALITY)
        self.assertTrue((f['restart_candidate'][65:85] == 1).all())


class StopStateTests(unittest.TestCase):
    def test_outside_confirmation_not_one_frame_noise(self):
        f = evidence()
        f['outside_margin_m'][5] = 1
        f['outside_margin_m'][30:50] = 1
        mask, _, _ = decode_stops(f, np.ones(250), 25, CONFIG, stationary=False)
        self.assertFalse(mask[5])
        self.assertTrue(mask[30:50].all())

    def test_still_ball_requires_context_and_tags_cannot_force_dead(self):
        f = evidence(); f['robust_ball_span_m'][:] = .2
        mask, _, _ = decode_stops(f, np.ones(250), 25, CONFIG, tag_windows=np.ones(250, bool))
        self.assertFalse(mask.any())

    def test_pressure_guard_rejects_duel_candidate(self):
        f = evidence(); f['robust_ball_span_m'][:] = .2
        f['quiet_fraction'][:] = 1; f['pressure_fraction'][:] = .5
        unguarded, _, _ = decode_stops(f, np.ones(250), 25, CONFIG)
        guarded, _, _ = decode_stops(f, np.ones(250), 25, CONFIG, pressure_guard=True)
        self.assertTrue(unguarded.all())
        self.assertFalse(guarded.any())

    def test_dead_memory_is_bounded_and_restart_clears_it(self):
        f = evidence(); f['outside_margin_m'][10:20] = 1
        held, _, _ = decode_stops(f, np.ones(250), 25, CONFIG, stationary=False, persistent=True)
        self.assertTrue(held[100]); self.assertFalse(held[160])
        f['restart_candidate'][50:70] = 1
        restarted, _, _ = decode_stops(f, np.ones(250), 25, CONFIG, stationary=False, persistent=True)
        self.assertTrue(restarted[49]); self.assertFalse(restarted[50:].any())

    def test_missing_frames_do_not_claim_ball_travel(self):
        m = Match(**vars(sample(150)))
        m.ball_xy_m[60:70] = np.nan
        m.ball_xy_m[70:, 0] += 20
        f = build_evidence(m, CONFIG, QUALITY)
        self.assertTrue(np.isnan(f['travel_speed_mps'][70:85]).all())


if __name__ == '__main__':
    unittest.main()
