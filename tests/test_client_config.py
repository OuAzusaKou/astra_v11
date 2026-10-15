import unittest
from pathlib import Path

from utils.client_config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_client_config


class ClientConfigTest(unittest.TestCase):
    def test_default_config_contains_no_machine_specific_absolute_paths(self):
        config = load_client_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config["policy"]["host"], "127.0.0.1")
        self.assertEqual(config["policy"]["port"], 29551)
        self.assertIsNone(config["startup"]["state_path"])
        self.assertIsNone(config["startup"]["reset_state_path"])
        self.assertEqual(config["runtime"]["post_chunk_sleep"], 1.5)
        self.assertEqual(
            Path(config["runtime"]["record_head"]),
            (REPO_ROOT / "records" / "live_infer.mp4").resolve(),
        )

    def test_pickplace_q20_profiles_use_distinct_ports_and_expected_steps(self):
        expected = {
            "client_q20_abs.yaml": (29551, 29565, 10, 10),
            "client_q20_abs_is5.yaml": (29552, 29563, 5, 5),
            "client_q20_abs_image5_action10.yaml": (29553, 29564, 10, 5),
        }

        for filename, values in expected.items():
            with self.subTest(filename=filename):
                config = load_client_config(REPO_ROOT / "config" / filename)
                local_port, remote_port, total_steps, image_steps = values
                self.assertEqual(config["policy"]["port"], local_port)
                self.assertEqual(config["tunnel"]["local_port"], local_port)
                self.assertEqual(config["tunnel"]["remote_port"], remote_port)
                self.assertEqual(
                    config["policy"]["expected_num_inference_steps"], total_steps
                )
                self.assertEqual(
                    config["policy"]["expected_image_num_inference_steps"],
                    image_steps,
                )

    def test_dexbench2_mano_profile_uses_dedicated_tunnel_and_mano_decode(self):
        config = load_client_config(
            REPO_ROOT / "config" / "client_dexbench2_mano.yaml"
        )

        self.assertEqual(config["policy"]["port"], 29555)
        self.assertEqual(config["policy"]["expected_num_inference_steps"], 10)
        self.assertEqual(config["tunnel"]["ssh_target"], "bce_dev01")
        self.assertEqual(config["tunnel"]["local_port"], 29555)
        self.assertEqual(config["tunnel"]["remote_port"], 29566)
        self.assertEqual(config["runtime"]["chunk_size"], 32)
        self.assertEqual(config["action"]["arm_type"], "relative")
        self.assertEqual(config["action"]["hand_type"], "relative")
        self.assertEqual(config["action"]["hand_source"], "mano")
        self.assertIsNone(config["startup"]["state_path"])
        self.assertIsNone(config["startup"]["reset_state_path"])

    def test_dex_bench_task_4_mano_profile_uses_dedicated_tunnel(self):
        config = load_client_config(
            REPO_ROOT / "config" / "client_dex_bench_task_4_mano.yaml"
        )

        self.assertEqual(config["policy"]["port"], 29557)
        self.assertEqual(config["policy"]["expected_num_inference_steps"], 10)
        self.assertEqual(config["tunnel"]["ssh_target"], "bce_dev01")
        self.assertEqual(config["tunnel"]["local_port"], 29557)
        self.assertEqual(config["tunnel"]["remote_port"], 29569)
        self.assertEqual(config["runtime"]["chunk_size"], 32)
        self.assertEqual(config["runtime"]["discard_steps"], 0)
        self.assertEqual(config["runtime"]["post_chunk_sleep"], 1.5)
        self.assertEqual(config["action"]["arm_type"], "relative")
        self.assertEqual(config["action"]["hand_type"], "relative")
        self.assertEqual(config["action"]["hand_source"], "mano")
        self.assertIsNone(config["startup"]["state_path"])
        self.assertIsNone(config["startup"]["reset_state_path"])

    def test_long_horizon_bench_1_v2_mano_profile_uses_dedicated_tunnel(self):
        config = load_client_config(
            REPO_ROOT / "config" / "client_long_horizon_bench_1_v2_mano.yaml"
        )

        self.assertEqual(config["policy"]["port"], 29558)
        self.assertEqual(config["policy"]["expected_num_inference_steps"], 10)
        self.assertEqual(config["tunnel"]["ssh_target"], "bce_dev01")
        self.assertEqual(config["tunnel"]["local_port"], 29558)
        self.assertEqual(config["tunnel"]["remote_port"], 29570)
        self.assertEqual(config["runtime"]["chunk_size"], 32)
        self.assertEqual(config["action"]["arm_type"], "relative")
        self.assertEqual(config["action"]["hand_type"], "relative")
        self.assertEqual(config["action"]["hand_source"], "mano")
        self.assertIsNone(config["startup"]["state_path"])
        self.assertIsNone(config["startup"]["reset_state_path"])

    def test_long_horizon_language_profiles_use_distinct_servers(self):
        expected = {
            "client_long_horizon_bench_1_v2_mano_l3.yaml": (29660, 29670),
            "client_long_horizon_bench_1_v2_mano_l2.yaml": (29661, 29671),
            "client_long_horizon_bench_1_v2_mano_l3_l2.yaml": (29662, 29672),
        }

        for filename, (local_port, remote_port) in expected.items():
            with self.subTest(filename=filename):
                config = load_client_config(REPO_ROOT / "config" / filename)
                self.assertEqual(config["policy"]["port"], local_port)
                self.assertEqual(config["policy"]["expected_num_inference_steps"], 10)
                self.assertEqual(config["tunnel"]["ssh_target"], "bce_dev01")
                self.assertEqual(config["tunnel"]["local_port"], local_port)
                self.assertEqual(config["tunnel"]["remote_port"], remote_port)
                self.assertEqual(config["runtime"]["chunk_size"], 32)
                self.assertEqual(config["runtime"]["post_chunk_sleep"], 1.5)
                self.assertIsNone(config["runtime"]["record_head"])
                self.assertEqual(config["action"]["arm_type"], "relative")
                self.assertEqual(config["action"]["hand_type"], "relative")
                self.assertEqual(config["action"]["hand_source"], "mano")


if __name__ == "__main__":
    unittest.main()
