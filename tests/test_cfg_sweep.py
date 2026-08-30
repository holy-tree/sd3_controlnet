import unittest

from scripts.cfg_sweep_test import _candidate_noise_indices, _pick_top_records


class CfgSweepRecordTest(unittest.TestCase):
    def test_selected_records_receive_stable_global_indices(self):
        records = [
            {"weather": "snow", "pair_id": "snow-b"},
            {"weather": "rain", "pair_id": "rain-b"},
            {"weather": "haze", "pair_id": "haze-a"},
            {"weather": "rain", "pair_id": "rain-a"},
            {"weather": "snow", "pair_id": "snow-a"},
        ]

        selected = _pick_top_records(records, samples_per_weather=1)

        self.assertEqual(
            [(row["weather"], row["pair_id"]) for row in selected],
            [("rain", "rain-a"), ("snow", "snow-a"), ("haze", "haze-a")],
        )
        self.assertEqual([row["global_index"] for row in selected], [0, 1, 2])

    def test_cfg_candidates_can_use_distinct_or_shared_noise(self):
        self.assertEqual(_candidate_noise_indices(6, False), [0, 1, 2, 3, 4, 5])
        self.assertEqual(_candidate_noise_indices(6, True), [0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
