from dataclasses import fields
import json
from pathlib import Path
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
                "feature_revalidate_radius": 4,
            },
            "hierarchical_feat05_grid0": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
            },
            "hierarchical_feat05_iterations2": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "n_iterations": 2,
            },
            "hierarchical_feat05_grid0_iterations2": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "n_iterations": 2,
            },
            "hierarchical_full_grid0": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full_iterations2": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "n_iterations": 2,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full_grid0_iterations2": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "n_iterations": 2,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
            },
            "hierarchical_full_grid0_mamba": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "mamba_refiner": True,
            },
            "hierarchical_full_grid0_mamba_replacement": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "mamba_replace_time_attention": True,
            },
            "hierarchical_full_grid0_iterations2_mlp_gate": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "n_iterations": 2,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "long_fusion_gate": "mlp",
            },
            "hierarchical_full_grid0_iterations2_mamba_gate": {
                "hierarchical_span": 10,
                "original_feature_weight": 0.5,
                "support_grid_size": 0,
                "n_iterations": 2,
                "occlusion_merge": True,
                "dual_anchor_weight": 0.5,
                "long_fusion_gate": "mamba",
            },
        }
        self.assertEqual(set(HIERARCHICAL_EXPERIMENTS), set(expected))
        names = [field.name for field in fields(ExperimentConfig)]
        for profile, changes in expected.items():
            config = HIERARCHICAL_EXPERIMENTS[profile]
            actual = {
                name: getattr(config, name)
                for name in names
                if getattr(config, name) != getattr(BASELINE, name)
            }
            self.assertEqual(actual, changes, profile)

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

    def test_mamba_strategies_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one Mamba"):
            ExperimentConfig(
                mamba_refiner=True,
                mamba_replace_time_attention=True,
            )

    def test_long_fusion_gate_requires_a_dual_anchor_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires dual anchor"):
            ExperimentConfig(long_fusion_gate="mamba")
        with self.assertRaisesRegex(ValueError, "must be none, mlp, or mamba"):
            ExperimentConfig(long_fusion_gate="transformer")

    def test_long_fusion_protocol_uses_real_labels_and_no_teacher(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "long-fusion-mamba-40-10-38.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["base_profile"], "hierarchical_full_grid0_iterations2")
        self.assertEqual(manifest["supervision"]["source"], "per-frame TrackRAD ground-truth segmentation")
        self.assertIsNone(manifest["supervision"]["teacher_model"])
        self.assertFalse(manifest["supervision"]["soft_confidence_labels"])
        self.assertEqual(
            [control["name"] for control in manifest["controls"]],
            [
                "hierarchical_full_grid0_iterations2",
                "oracle_branch_selection",
                "hierarchical_full_grid0_iterations2_mlp_gate",
                "hierarchical_full_grid0_iterations2_mamba_gate",
            ],
        )

    def test_mamba_protocol_is_frozen_to_40_10_38(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "mamba-40-10-38.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        profiles = [entry["name"] for entry in manifest["profiles"]]

        self.assertEqual(
            (protocol["training_cases"], protocol["validation_cases"], protocol["test_cases"]),
            (40, 10, 38),
        )
        self.assertEqual(protocol["auxiliary_teacher_weight"], 0.0)
        self.assertEqual(protocol["confidence_target"], "hard")
        self.assertEqual(
            profiles,
            [
                "hierarchical_full_grid0_mamba",
                "hierarchical_full_grid0_mamba_replacement",
            ],
        )
        self.assertTrue(set(profiles) <= set(HIERARCHICAL_EXPERIMENTS))
        self.assertEqual(
            manifest["reporting"]["primary_reference"],
            "hierarchical_full_grid0",
        )

    def test_official_experiment_registry_is_consistent(self) -> None:
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "experiment-status.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        official = set(registry["official_combination_profiles"])
        candidates = set(registry["official_candidates"].values())
        excluded = set(registry["exploratory_excluded"])

        self.assertEqual(official, {
            "hierarchical_feat05_grid0",
            "hierarchical_feat05_iterations2",
            "hierarchical_feat05_grid0_iterations2",
            "hierarchical_full_grid0",
            "hierarchical_full_iterations2",
            "hierarchical_full_grid0_iterations2",
        })
        self.assertTrue(candidates <= official)
        self.assertTrue(official <= set(HIERARCHICAL_EXPERIMENTS))
        self.assertFalse(official & excluded)
        self.assertFalse(registry["policy"]["include_exploratory_in_official_ranking"])

    def test_public_test_queue_is_frozen_and_valid(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "public-test-38-profiles.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        profiles = [entry["name"] for entry in manifest["profiles"]]

        self.assertEqual(manifest["expected_cases"], 38)
        self.assertEqual(profiles[0], "baseline")
        self.assertEqual(len(profiles), 12)
        self.assertEqual(len(profiles), len(set(profiles)))
        self.assertTrue(set(profiles) <= set(EXPERIMENTS))
        self.assertEqual(
            profiles,
            [
                "baseline",
                "points_500",
                "support_grid_0",
                "iterations_2",
                "grid0_iterations2",
                "grid0_stride2",
                "hierarchical_d10_original_feat_05",
                "hierarchical_full",
                "hierarchical_feat05_grid0",
                "hierarchical_feat05_grid0_iterations2",
                "hierarchical_full_grid0_iterations2",
                "hierarchical_full_grid0",
            ],
        )

    def test_remaining_public_test_queue_covers_unreported_profiles(self) -> None:
        experiment_dir = Path(__file__).resolve().parents[1] / "experiments"
        reported = json.loads(
            (experiment_dir / "public-test-38-profiles.json").read_text(
                encoding="utf-8"
            )
        )
        remaining = json.loads(
            (experiment_dir / "public-test-38-remaining-profiles.json").read_text(
                encoding="utf-8"
            )
        )
        reported_names = {entry["name"] for entry in reported["profiles"]}
        remaining_names = [entry["name"] for entry in remaining["profiles"]]
        excluded_names = {entry["name"] for entry in remaining["excluded"]}

        self.assertEqual(remaining["expected_cases"], 38)
        self.assertEqual(len(remaining_names), len(set(remaining_names)))
        self.assertFalse(reported_names & set(remaining_names))
        self.assertEqual(
            set(remaining_names),
            set(EXPERIMENTS) - reported_names - excluded_names,
        )
        self.assertEqual(remaining_names[-1], "support_grid_15")
        self.assertEqual(
            excluded_names,
            {
                "points_1500",
                "hierarchical_full_grid0_mamba",
                "hierarchical_full_grid0_mamba_replacement",
                "hierarchical_full_grid0_iterations2_mlp_gate",
                "hierarchical_full_grid0_iterations2_mamba_gate",
            },
        )

    def test_random_tutor_test_manifest_forbids_soft_labels(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "random-tutor-hard-label-test-38.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = [entry["name"] for entry in manifest["profiles"]]

        self.assertEqual(manifest["training_cases"], 40)
        self.assertEqual(manifest["evaluation_cases"], 38)
        self.assertEqual(manifest["confidence_target_mode"], "hard")
        self.assertFalse(manifest["soft_confidence_labels"])
        self.assertEqual(manifest["seeds"], [0, 1, 2])
        self.assertEqual(
            names,
            [
                "baseline_single_teacher",
                "random_tutor_w010",
                "random_tutor_w015",
                "random_tutor_w020",
                "random_tutor_w025",
                "same_teacher_control_w020",
            ],
        )
        training_runner = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_random_tutor_training_resumable.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('"--confidence_target_mode", "hard"', training_runner)


if __name__ == "__main__":
    unittest.main()
