from dataclasses import fields
import unittest

from experiments import (
    AUDIT_EXPERIMENTS,
    BASELINE,
    COMBINATION_EXPERIMENTS,
    EXPERIMENTS,
    HIERARCHICAL_EXPERIMENTS,
    SINGLE_POINT_EXPERIMENTS,
    ExperimentConfig,
    get_experiment_config,
)


class ExperimentTests(unittest.TestCase):
    def test_baseline_matches_original_submission(self) -> None:
        self.assertEqual(BASELINE, ExperimentConfig())
        self.assertEqual(BASELINE.border_points, 1000)
        self.assertEqual(BASELINE.support_grid_size, 10)
        self.assertEqual(BASELINE.n_iterations, 4)

    def test_each_profile_changes_exactly_one_field(self) -> None:
        names = [field.name for field in fields(ExperimentConfig)]
        for profile, config in SINGLE_POINT_EXPERIMENTS.items():
            if profile == "baseline":
                continue
            changed = [
                name
                for name in names
                if getattr(config, name) != getattr(BASELINE, name)
            ]
            self.assertEqual(len(changed), 1, (profile, changed))

    def test_combination_profiles_change_only_supported_fields(self) -> None:
        expected = {
            "grid0_iterations2": {"support_grid_size", "n_iterations"},
            "grid0_stride2": {"support_grid_size", "temporal_stride"},
            "grid0_iterations2_stride2": {
                "support_grid_size",
                "n_iterations",
                "temporal_stride",
            },
        }
        names = [field.name for field in fields(ExperimentConfig)]
        self.assertEqual(set(COMBINATION_EXPERIMENTS), set(expected))
        for profile, config in COMBINATION_EXPERIMENTS.items():
            changed = {
                name
                for name in names
                if getattr(config, name) != getattr(BASELINE, name)
            }
            self.assertEqual(changed, expected[profile], profile)

    def test_all_profiles_are_available(self) -> None:
        self.assertEqual(
            set(EXPERIMENTS),
            set(SINGLE_POINT_EXPERIMENTS)
            | set(COMBINATION_EXPERIMENTS)
            | set(HIERARCHICAL_EXPERIMENTS)
            | set(AUDIT_EXPERIMENTS),
        )

    def test_audit_baseline_repeat_is_an_exact_duplicate(self) -> None:
        self.assertEqual(AUDIT_EXPERIMENTS, {"baseline_repeat": BASELINE})

    def test_hierarchical_profiles_form_the_requested_ablation_series(self) -> None:
        expected = {
            "hierarchical_d10": {"hierarchical_span": 10},
            "hierarchical_d10_original_feat_05": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
            },
            "hierarchical_d10_occlusion_merge": {
                "hierarchical_span": 10,
                "occlusion_merge": True,
            },
            "hierarchical_d10_dual_anchor": {
                "hierarchical_span": 10,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_d10_original_feat_025": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.25,
            },
            "hierarchical_d10_original_feat_075": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.75,
            },
            "hierarchical_full_d5": {
                "hierarchical_span": 5,
                "original_feature_weight": 0.5,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full_d15": {
                "hierarchical_span": 15,
                "original_feature_weight": 0.5,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full_feature_gate": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "feature_gate_distance": 4.0,
            },
            "hierarchical_full_feature_revalidate_r4": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "feature_gate_distance": 4.0,
                "feature_similarity_threshold": 0.5,
                "feature_revalidate_radius": 4,
            },
        }
        self.assertEqual(set(HIERARCHICAL_EXPERIMENTS), set(expected))
        for profile, changes in expected.items():
            config = HIERARCHICAL_EXPERIMENTS[profile]
            for name, value in changes.items():
                self.assertEqual(getattr(config, name), value, (profile, name))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown COTRACKER_EXPERIMENT"):
            get_experiment_config("does-not-exist")

    def test_feature_revalidation_requires_gate_and_dual_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires dual anchor"):
            ExperimentConfig(
                hierarchical_span=10,
                original_feature_weight=0.5,
                feature_gate_distance=4.0,
            )
        with self.assertRaisesRegex(ValueError, "requires feature gating"):
            ExperimentConfig(
                hierarchical_span=10,
                original_feature_weight=0.5,
                dual_anchor_weight=0.5,
                feature_revalidate_radius=4,
            )


if __name__ == "__main__":
    unittest.main()
