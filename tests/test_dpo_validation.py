import unittest

from dpo.validation import (
    summarize_validation_rows,
    validation_prompt_for_record,
)


class DpoValidationSummaryTest(unittest.TestCase):
    def test_reports_each_weather_and_weighted_overall_metrics(self):
        rows = [
            {"weather": "rain", "psnr": 20.0, "ssim": 0.8, "lpips": 0.2},
            {"weather": "rain", "psnr": 22.0, "ssim": 0.9, "lpips": 0.1},
            {"weather": "snow", "psnr": 30.0, "ssim": 0.7, "lpips": 0.3},
        ]

        per_weather, overall = summarize_validation_rows(
            rows, ["rain", "snow", "haze"]
        )

        self.assertEqual(per_weather["rain"]["n"], 2)
        self.assertEqual(per_weather["rain"]["psnr"], 21.0)
        self.assertEqual(per_weather["snow"]["psnr"], 30.0)
        self.assertNotIn("haze", per_weather)
        self.assertAlmostEqual(overall["psnr"], 24.0)
        self.assertEqual(overall["n"], 3)

    def test_validation_prompt_uses_weather_mapping(self):
        config = {
            "use_prompt": True,
            "weather_prompts": {"rain": "clear rain-free image"},
        }
        self.assertEqual(
            validation_prompt_for_record({"weather": "rain"}, config),
            "clear rain-free image",
        )
        self.assertEqual(
            validation_prompt_for_record({"weather": "rain"}, {**config, "use_prompt": False}),
            "",
        )


if __name__ == "__main__":
    unittest.main()
