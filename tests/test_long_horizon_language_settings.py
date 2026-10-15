import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = ROOT / "ops/long_horizon_bench_1_v2_language_mano"
GEMINI35_OPS_DIR = ROOT / "ops/long_horizon_bench_1_v2_gemini35_mano"


def _load_selector():
    spec = importlib.util.spec_from_file_location(
        "long_horizon_select_setting", OPS_DIR / "select_setting.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LongHorizonLanguageSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = _load_selector()
        cls.manifest = json.loads(
            (OPS_DIR / "settings.json").read_text(encoding="utf-8")
        )

    def test_manifest_has_all_24_settings_and_unique_l3(self):
        settings = self.manifest["settings"]
        self.assertEqual(self.manifest["episode_count"], 1206)
        self.assertEqual(self.manifest["setting_count"], 24)
        self.assertEqual(len(settings), 24)
        self.assertEqual(len({value["l3_zh"] for value in settings.values()}), 24)
        expected = {
            f"{task}_c{combination}"
            for task in (
                "cleartable1",
                "cleartable3",
                "preparebreakfast1",
                "preparebreakfast4",
            )
            for combination in range(1, 7)
        }
        self.assertEqual(set(settings), expected)

    def test_all_init_states_are_finite_and_shaped(self):
        widths = {
            "left_ee_pose": 6,
            "right_ee_pose": 6,
            "left_hand_state": 20,
            "right_hand_state": 20,
        }
        for setting in self.manifest["settings"].values():
            for field, width in widths.items():
                value = np.asarray(setting["mean_by_key"][field])
                self.assertEqual(value.shape, (width,))
                self.assertTrue(np.isfinite(value).all())

    def test_all_72_prompt_plans_match_training_contract(self):
        for setting in self.manifest["settings"].values():
            self.assertEqual(self.selector._prompts(setting, "l3"), [setting["l3_en"]])
            self.assertEqual(self.selector._prompts(setting, "l2"), setting["l2"])
            l3 = setting["l3_zh"].strip().rstrip(".。!?！？;；") + "."
            self.assertEqual(
                self.selector._prompts(setting, "l3_l2"),
                [f"{l3} {value}" for value in setting["l2"]],
            )


class Gemini35LongHorizonLanguageSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = _load_selector()
        cls.manifest = json.loads(
            (GEMINI35_OPS_DIR / "settings.json").read_text(encoding="utf-8")
        )

    def test_all_settings_use_group_reviewed_instruction_sources(self):
        settings = self.manifest["settings"]
        self.assertEqual(len(settings), 24)
        self.assertEqual(
            sum(value["instruction_reviewed_episode_count"] for value in settings.values()),
            1206,
        )
        for setting in settings.values():
            self.assertIn("comparing every episode", setting["instruction_selection_method"])
            self.assertGreater(setting["selected_stage_count_frequency"], 0)
            self.assertNotIn("retract the hand", setting["l2"])
            self.assertNotIn("return the hand to the initial position", setting["l2"])
            video_path = Path(self.manifest["dataset_root"]) / setting["demo_video_path"]
            self.assertTrue(video_path.is_file())
            self.assertIn("observation.images.image_ego_0", setting["demo_video_path"])

    def test_cleartable3_combination6_names_concrete_objects(self):
        setting = self.manifest["settings"]["cleartable3_c6"]
        self.assertEqual(setting["instruction_source_episode_index"], 1134)
        self.assertEqual(setting["instruction_source_collect_id"], "1787928733_5")
        self.assertEqual(setting["instruction_reviewed_episode_count"], 51)
        self.assertEqual(
            setting["l2"][:2],
            [
                "pick up the orange and throw it into the trash can",
                "pick up the crumpled tissue and throw it into the trash can",
            ],
        )
        self.assertEqual(
            setting["l2_zh"][:2],
            ["拿起橘子并扔进垃圾桶", "拿起揉皱的纸巾并扔进垃圾桶"],
        )

    def test_all_72_prompt_plans_use_the_checked_manifest(self):
        for setting in self.manifest["settings"].values():
            self.assertEqual(self.selector._prompts(setting, "l3"), [setting["l3_en"]])
            self.assertEqual(self.selector._prompts(setting, "l2"), setting["l2"])
            l3 = setting["l3_zh"].strip().rstrip(".。!?！？;；") + "."
            self.assertEqual(
                self.selector._prompts(setting, "l3_l2"),
                [f"{l3} {value}" for value in setting["l2"]],
            )

    def test_runbook_links_the_separate_result_spreadsheet(self):
        runbook = (
            ROOT
            / "docs/long-horizon-bench-1-v2-gemini35-l2-l3l2-mano-deployment.md"
        ).read_text(encoding="utf-8")
        result_section = runbook.split("## 3. 结果记录", 1)[1]
        self.assertIn(
            "https://ucn81zkkano6.feishu.cn/sheets/"
            "IUnKsShjdhEwcNtcu7rcrv9Lnqc",
            result_section,
        )
        self.assertNotIn("| 任务 | Combination |", result_section)

    def test_runbook_links_one_training_video_per_setting(self):
        runbook = (
            ROOT
            / "docs/long-horizon-bench-1-v2-gemini35-l2-l3l2-mano-deployment.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(runbook.count("训练示范视频（主视角，评测前先完整观看）"), 24)

    def test_runbook_lists_three_trials_in_model_interleaved_order(self):
        runbook = (
            ROOT
            / "docs/long-horizon-bench-1-v2-gemini35-l2-l3l2-mano-deployment.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(runbook.count("Trial 1："), 24)
        self.assertEqual(runbook.count("Trial 2："), 24)
        self.assertEqual(runbook.count("Trial 3："), 24)
        for setting in self.manifest["settings"]:
            for trial in range(1, 4):
                ordered = "\n".join(
                    [
                        f"long_horizon_bench_1_v2_eval_l3_{setting} {trial}",
                        f"long_horizon_bench_1_v2_gemini35_eval_l2_{setting} {trial}",
                        f"long_horizon_bench_1_v2_gemini35_eval_l3_l2_{setting} {trial}",
                    ]
                )
                self.assertIn(ordered, runbook)

    def test_runbook_documents_start_image_and_subtask_result_formats(self):
        runbook = (
            ROOT
            / "docs/long-horizon-bench-1-v2-gemini35-l2-l3l2-mano-deployment.md"
        ).read_text(encoding="utf-8")
        self.assertIn("[start-image] saved -> ...", runbook)
        self.assertIn("setting、trial 和模型", runbook)
        self.assertIn("1/1/1/1/1/1/0", runbook)

    def test_eval_alias_requires_trial_and_passes_start_image_path(self):
        alias_script = (GEMINI35_OPS_DIR / "alias.sh").read_text(encoding="utf-8")
        self.assertIn("usage: ${command_name} TRIAL [reviewed-init-state]", alias_script)
        self.assertIn('--start-image-path "${start_image_path}"', alias_script)
        self.assertIn("trial-${trial}", alias_script)
        self.assertIn('LONG_HORIZON_BENCH_1_V2_ROS_DOMAIN_ID:-105', alias_script)


if __name__ == "__main__":
    unittest.main()
