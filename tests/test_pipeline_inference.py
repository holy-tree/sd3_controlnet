import types
import unittest

import torch

from utils.pipeline_inference import run_pipeline_with_fp32_decode


class _FakeVae:
    dtype = torch.float32
    config = types.SimpleNamespace(scaling_factor=2.0, shift_factor=0.25)

    def __init__(self):
        self.decoded_latents = None

    def decode(self, latents, return_dict=False):
        self.decoded_latents = latents
        return (latents[:, :3],)


class _FakeImageProcessor:
    @staticmethod
    def postprocess(images, output_type):
        return [images]


class _FakePipeline:
    def __init__(self):
        self.vae = _FakeVae()
        self.image_processor = _FakeImageProcessor()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        latents = torch.full((1, 4, 2, 2), 2.0)
        return types.SimpleNamespace(images=latents)


class PipelineInferenceTest(unittest.TestCase):
    def test_decodes_latent_in_fp32_after_mixed_precision_denoising(self):
        pipeline = _FakePipeline()

        images = run_pipeline_with_fp32_decode(
            pipeline,
            {"prompt": "test"},
            device=torch.device("cpu"),
            denoise_dtype=torch.bfloat16,
        )

        self.assertEqual(pipeline.kwargs["output_type"], "latent")
        self.assertEqual(pipeline.vae.decoded_latents.dtype, torch.float32)
        torch.testing.assert_close(
            pipeline.vae.decoded_latents,
            torch.full((1, 4, 2, 2), 1.25),
        )
        self.assertEqual(len(images), 1)


if __name__ == "__main__":
    unittest.main()
