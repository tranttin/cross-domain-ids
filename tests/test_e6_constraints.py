from __future__ import annotations

import numpy as np
import unittest

from ids.robustness.constraints import build_semantic_projector, resolve_mutable_coordinates


MUTABLE_2018 = [
    "Bwd Pkts/s",
    "Flow Pkts/s",
    "Fwd Pkts/s",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Min",
    "Pkt Len Std",
]


class IdentityScaler:
    def __init__(self, width):
        self.center_ = np.zeros(width, dtype=np.float32)
        self.scale_ = np.ones(width, dtype=np.float32)


def _projector():
    names = [*MUTABLE_2018, "Immutable Feature"]
    source = np.vstack([
        np.full(len(names), -10.0, dtype=np.float32),
        np.full(len(names), 10.0, dtype=np.float32),
    ])
    projector, cfg = build_semantic_projector(
        selected_feature_names=names,
        scaler=IdentityScaler(len(names)),
        source_train_scaled=source,
        dataset_name="2018",
        constraint_path="configs/robustness/e6_constraints_v051.yaml",
        canonical_mapping_path="configs/features/canonical_features.yaml",
    )
    return projector, cfg


class E6ConstraintTests(unittest.TestCase):
    def test_every_mutable_semantic_feature_resolves_once_for_2018(self):
        projector, cfg = _projector()
        self.assertEqual(set(projector.semantic_to_index), set(cfg["mutable_semantic_features"]))
        self.assertEqual(len(set(projector.semantic_to_index.values())), 7)
        self.assertEqual(int(projector.mutable_mask.sum()), 7)

    def test_projector_enforces_budget_immutability_and_raw_relations(self):
        projector, _ = _projector()
        # Order in the model is max, mean, min. The clean row is feasible but close
        # enough that a one-unit attack can invert all three statistics.
        x0 = np.asarray([[0.2, 0.3, 0.1, 5.1, 5.0, 4.9, 0.2, 7.0]], dtype=np.float32)
        candidate = np.asarray([[-2.0, -2.0, -2.0, 4.1, 4.0, 5.9, -1.0, -99.0]], dtype=np.float32)
        projected = projector.project(candidate, x0, 1.0)
        audit = projector.audit(projected, x0, 1.0)
        self.assertTrue(audit["constraint_pass"])
        self.assertEqual(audit["constraint_violation_count"], 0)
        self.assertAlmostEqual(float(projected[0, -1]), float(x0[0, -1]))
        self.assertLessEqual(float(np.max(np.abs(projected - x0))), 1.0 + 1e-6)
        self.assertTrue(np.all(projected[0, :7] >= -1e-6))
        self.assertLessEqual(projected[0, 5], projected[0, 4])
        self.assertLessEqual(projected[0, 4], projected[0, 3])

    def test_projector_preserves_cnn_channel_shape(self):
        projector, _ = _projector()
        x0 = np.asarray([[0.2, 0.3, 0.1, 5.1, 5.0, 4.9, 0.2, 7.0]], dtype=np.float32)[..., None]
        projected = projector.project(x0 + 0.2, x0, 0.2)
        self.assertEqual(projected.shape, x0.shape)
        self.assertTrue(projector.audit(projected, x0, 0.2)["constraint_pass"])

    def test_preexisting_order_violation_is_frozen_not_silently_repaired(self):
        projector, _ = _projector()
        # min (index 5) is above mean (index 4): this inconsistency belongs to
        # the clean dataset and cannot be repaired without changing evaluation.
        x0 = np.asarray([[0.2, 0.3, 0.1, 5.1, 4.8, 4.9, 0.2, 7.0]], dtype=np.float32)
        candidate = x0.copy()
        candidate[0, 3:6] += np.asarray([-0.5, 0.5, 0.5], dtype=np.float32)
        projected = projector.project(candidate, x0, 0.5)
        self.assertTrue(np.allclose(projected[0, 3:6], x0[0, 3:6]))
        audit = projector.audit(projected, x0, 0.5)
        self.assertTrue(audit["constraint_pass"])
        self.assertEqual(audit["constraint_violation_count"], 0)
        self.assertEqual(audit["clean_ordered_triplet_violation_count"], 1)
        self.assertEqual(audit["absolute_ordered_triplet_violation_count"], 1)
        self.assertEqual(audit["ordered_triplet_violation_count"], 0)
        self.assertEqual(audit["preexisting_ordered_triplet_modified_count"], 0)
        self.assertTrue(projector.assert_clean_feasible(x0)["constraint_pass"])

    def test_missing_mutable_coordinate_is_a_hard_failure(self):
        with self.assertRaisesRegex(ValueError, "exactly one coordinate"):
            resolve_mutable_coordinates(
                MUTABLE_2018[:-1], ["packet_length_std"], dataset_name="2018",
                canonical_mapping_path="configs/features/canonical_features.yaml",
            )


if __name__ == "__main__":
    unittest.main()
