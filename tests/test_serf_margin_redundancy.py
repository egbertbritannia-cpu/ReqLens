"""CPU-only mathematical contract for the legacy SERF confidence/margin router.

IMPLEMENTED_AND_TESTED_ON_SYNTHETIC_PROBABILITIES; not a model experiment.
The two notebooks calculate confidence=max(softmax logits),
margin=largest_probability-second_largest_probability, then route when
confidence < tau OR margin < delta OR predicted_class in risky_set.

Run: python -m unittest discover -s tests -p 'test_serf_margin_redundancy.py' -v
No third-party dependencies, model inference, downloads, or training.
"""
import unittest

TAUS = (0.70, 0.80, 0.85, 0.90)
DELTAS = (0.15, 0.20)


def signals(probabilities, tau, delta, risky_labels=()):
    """Reference expression only; notebook implementations remain authoritative."""
    if len(probabilities) < 2 or any(p < 0 for p in probabilities):
        raise ValueError("Expected a probability distribution with >=2 classes")
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("Probabilities must sum to 1")
    ranked = sorted(enumerate(probabilities), key=lambda pair: pair[1], reverse=True)
    confidence, runner_up = ranked[0][1], ranked[1][1]
    margin = confidence - runner_up
    c = confidence < tau
    m = margin < delta
    r = ranked[0][0] in risky_labels
    return c, m, r


class LegacyMarginRedundancyTests(unittest.TestCase):
    def test_analytic_margin_bound_for_every_legacy_parameter_pair(self):
        # p2 <= 1 - p1, so margin = p1 - p2 >= 2*p1 - 1.
        # If p1 >= tau, margin >= 2*tau - 1 >= 0.40 > delta.
        for tau in TAUS:
            for delta in DELTAS:
                with self.subTest(tau=tau, delta=delta):
                    self.assertGreater(2 * tau - 1, delta)

    def test_valid_simplex_never_has_margin_only_trigger(self):
        # Enumerate 5,151 *valid* 3-class distributions at 0.01 resolution.
        # An impossible combination like p1=0.90, p2=0.80 must NOT be used.
        for a in range(101):
            for b in range(101 - a):
                probs = (a / 100, b / 100, (100 - a - b) / 100)
                for tau in TAUS:
                    for delta in DELTAS:
                        c, m, _ = signals(probs, tau, delta)
                        self.assertFalse(m and not c)

    def test_removing_margin_does_not_change_full_legacy_route(self):
        for a in range(0, 101, 2):
            for b in range(0, 101 - a, 2):
                probs = (a / 100, b / 100, (100 - a - b) / 100)
                for tau in TAUS:
                    for delta in DELTAS:
                        for risky in ((), (0,), (1,), (2,)):
                            c, m, r = signals(probs, tau, delta, risky)
                            self.assertEqual(c or m or r, c or r)

    def test_margin_can_trigger_but_only_alongside_confidence(self):
        c, m, r = signals((0.55, 0.40, 0.05), 0.70, 0.20)
        self.assertEqual((c, m, r), (True, True, False))

    def test_margin_only_is_possible_outside_legacy_grid(self):
        c, m, r = signals((0.52, 0.40, 0.08), 0.50, 0.20)
        self.assertEqual((c, m, r), (False, True, False))

    def test_risky_class_can_independently_trigger_fusion(self):
        c, m, r = signals((0.90, 0.05, 0.05), 0.70, 0.20, risky_labels=(0,))
        self.assertEqual((c, m, r), (False, False, True))


if __name__ == "__main__":
    unittest.main()
