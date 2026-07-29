"""
成对图像数据集加载器 (SD3 ControlNet 适配版)
==============================================

迁移自 D:\\Projects\\pycharm\\controlnet_file\\dataloaders\\paired_dataset.py,
适配 Stable Diffusion 3 ControlNet 训练脚本 train_controlnet_sd3.py.

支持目录结构:
    dataset_root/
    ├── rain/{train,test}/{GT,LQ}/
    ├── snow/{train,test}/{GT,LQ}/
    └── haze/{train,test}/{GT,LQ}/

两套输出模式:
    defer_transforms=False (默认, 源项目行为):
        __getitem__ 内完成 Resize/CenterCrop/ToTensor/Normalize,
        返回:
            {
                "pixel_values":            tensor [3,H,W] in [-1, 1]   (GT)
                "conditioning_pixel_values": tensor [3,H,W] in [0, 1]  (LQ)
                "input_ids":              LongTensor [77]              (CLIP tokenized)
                "weather":                str
            }

    defer_transforms=True (SD3 HF imagefolder 流程):
        __getitem__ 只读取原始 PIL 图 + 解析 prompt 字符串, 不做任何变换.
        返回:
            {
                "image":              PIL.Image (RGB GT)         → 对应 HF image_column
                "conditioning_image": PIL.Image (RGB LQ)         → 对应 HF conditioning_image_column
                "text":               str (已经解析好的 prompt)  → 对应 HF caption_column
                "weather":            str
            }
        后续由 SD3 脚本里的 preprocess_train / with_transform 完成 Resize+CenterCrop+ToTensor+Normalize,
        再由 dataset.map(compute_embeddings_fn, batched=True) 做 3-编码器 prompt 预编码.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils import data as data
from torchvision import transforms

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 默认各天气的 prompt 描述 (与源项目一致)
DEFAULT_WEATHER_PROMPTS: Dict[str, str] = {
    "rain": "rainy scene, rain streaks on the image, wet surfaces, overcast sky",
    "snow": "snowy scene, snowflakes covering the image, cold atmosphere, white noise",
    "haze": "hazy scene, foggy atmosphere, low visibility, grayish tone",
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


class PairedCaptionDataset(data.Dataset):
    """
    从 dataset_root/{weather}/{split}/{GT,LQ}/ 加载图像对, 适配 SD3 ControlNet 训练.

    Args:
        dataset_root: 数据集根目录 (例如 D:/Projects/pycharm/WeaFU-main/dataprocess)
        weather_types: 天气类型列表, 例如 ['rain', 'snow', 'haze']
        splits: 划分列表, 例如 ['train']
        tokenizer: HuggingFace tokenizer (仅 defer_transforms=False 使用,
                   SD3 模式下应传 None, prompt 由上游 pre-tokenize)
        use_prompt: 是否使用 prompt 文本 (False: 全部空 prompt)
        prompt_ratio: 启用 prompt 时, 使用天气 prompt 的概率 (0.15 ~ 0.25)
        weather_prompts: 自定义各天气的 prompt 描述 (可选)
        resolution: 训练分辨率 (默认 512), 短边 Resize 后 CenterCrop
        weather_num_samples: 按天气类型限制样本数, 例如 {'rain': 10, 'snow': 5}
        defer_transforms: True = 返回 PIL + str (供 SD3 脚本的 with_transform/预处理流程);
                          False = 返回 tensor (源项目旧行为, 直接 collate)
    """

    def __init__(
        self,
        dataset_root: str = "",
        weather_types: List[str] = None,
        splits: List[str] = None,
        tokenizer=None,
        null_text_ratio: float = 0.5,
        use_prompt: bool = False,
        prompt_ratio: float = 0.2,
        weather_prompts: Dict[str, str] = None,
        resolution: int = 512,
        weather_num_samples: Dict[str, int] = None,
        defer_transforms: bool = False,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.null_text_ratio = null_text_ratio
        self.use_prompt = use_prompt
        self.prompt_ratio = max(0.0, min(1.0, prompt_ratio))
        self.resolution = resolution
        self.weather_num_samples = weather_num_samples or {}
        self.defer_transforms = defer_transforms

        if weather_types is None:
            weather_types = ["rain", "snow", "haze"]
        if splits is None:
            splits = ["train"]

        self.weather_types = list(weather_types)
        self.splits = list(splits)

        # 合并自定义与默认 prompt
        self.weather_prompts = dict(DEFAULT_WEATHER_PROMPTS)
        if weather_prompts:
            self.weather_prompts.update(weather_prompts)

        # 加载所有图像对: list of (gt_path, lq_path, weather)
        self.samples: List[Tuple[Path, Path, str]] = []
        for weather in self.weather_types:
            for split in self.splits:
                gt_dir = self.dataset_root / weather / split / "GT"
                lq_dir = self.dataset_root / weather / split / "LQ"
                if not gt_dir.is_dir() or not lq_dir.is_dir():
                    print(f"[跳过] {gt_dir} 或 {lq_dir} 不存在")
                    continue

                gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
                lq_map = {p.stem: p for p in lq_dir.iterdir() if p.is_file() and is_image(p)}

                matched = 0
                for stem in sorted(gt_map.keys() & lq_map.keys()):
                    self.samples.append((gt_map[stem], lq_map[stem], weather))
                    matched += 1

                print(f"[数据集] {weather}/{split}: 匹配 {matched} 对")

        if not self.samples:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下未找到任何匹配的图像对, "
                f"请检查目录结构是否为 {{weather}}/{{split}}/{{GT,LQ}}/"
            )

        # ===== 按 weather 限制每个天气的样本数 =====
        if self.weather_num_samples:
            grouped: Dict[str, List[Tuple[Path, Path, str]]] = {w: [] for w in self.weather_types}
            for sample in self.samples:
                if sample[2] in grouped:
                    grouped[sample[2]].append(sample)

            new_samples: List[Tuple[Path, Path, str]] = []
            for weather in self.weather_types:
                limit = self.weather_num_samples.get(weather, -1)
                weather_samples = grouped[weather]
                if limit is not None and limit > 0 and limit < len(weather_samples):
                    print(f"[数据集] {weather}: 截断 {len(weather_samples)} -> {limit} 样本")
                    new_samples.extend(weather_samples[:limit])
                else:
                    new_samples.extend(weather_samples)
            self.samples = new_samples

            print(f"[数据集] 最终训练样本数: {len(self.samples)}")
            for w in self.weather_types:
                cnt = sum(1 for s in self.samples if s[2] == w)
                limit_str = f"/{self.weather_num_samples[w]}" if self.weather_num_samples.get(w, -1) > 0 else ""
                print(f"  - {w}: {cnt}{limit_str}")

        # ===== 图像预处理 (仅 defer_transforms=False 模式使用) =====
        # SD3 训练脚本会在 preprocess_train 里自行应用同样的 transform, 故 defer 模式下不构建.
        if not self.defer_transforms:
            self.preprocess = transforms.Compose([
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
            ])
            self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # weather prompt 解析 (供 SD3 defer_transforms 模式使用, 让上游 process_captions
    # 在 proportion_empty_prompts=0 时变成 passthrough)
    #
    # deterministic=True 时不调 random, 用 hash(stem) 推一个稳定的 bool,
    # 保证预计算与 __getitem__ 取到一致结果 (避免"预编码用 prompt A, 取样用 prompt B")
    # ------------------------------------------------------------------
    def _make_prompt(self, weather: str, deterministic_seed: str | None = None) -> str:
        if not self.use_prompt:
            return ""
        if deterministic_seed is not None:
            import hashlib
            h = int(hashlib.md5(deterministic_seed.encode("utf-8")).hexdigest()[:8], 16) % (10 ** 6)
            use_weather_prompt = (h / 10 ** 6) < self.prompt_ratio
        else:
            use_weather_prompt = random.random() < self.prompt_ratio
        if use_weather_prompt:
            return self.weather_prompts.get(weather, "")
        return ""

    def attach_precomputed(self, prompt_embeds_list, pooled_prompt_embeds_list, resolved_prompts):
        """
        绑定预计算的 SD3 prompt embeddings 与解析好的 prompt 字符串.

        调用后, __getitem__ 返回:
            {
                "pixel_values":             tensor [-1, 1],
                "conditioning_pixel_values": tensor [0, 1],
                "prompt_embeds":             tensor (T5 长度, dim),
                "pooled_prompt_embeds":      tensor (dim,),
                "weather":                   str,
            }

        Args:
            prompt_embeds_list:     list[Tensor], 每个形状 [seq_len, joint_dim]
            pooled_prompt_embeds_list: list[Tensor], 每个形状 [joint_dim]
            resolved_prompts:       list[str], 与 samples 等长, 来自同一次预编码时的 _make_prompt
        """
        assert len(prompt_embeds_list) == len(self.samples), \
            f"prompt_embeds_list 长度 {len(prompt_embeds_list)} != samples {len(self.samples)}"
        assert len(pooled_prompt_embeds_list) == len(self.samples)
        assert len(resolved_prompts) == len(self.samples)
        self._prompt_embeds = prompt_embeds_list
        self._pooled_prompt_embeds = pooled_prompt_embeds_list
        self._resolved_prompts = resolved_prompts

    # ------------------------------------------------------------------
    # SD2 兼容: 直接做 CLIP tokenize (仅 defer_transforms=False 使用)
    # ------------------------------------------------------------------
    def tokenize_caption(self, caption: str = "") -> torch.Tensor:
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result = inputs.input_ids
        if not torch.is_tensor(result):
            result = torch.tensor(result)
        return result

    def __getitem__(self, index):
        gt_path, lq_path, weather = self.samples[index]

        # 若已 attach_precomputed, 用预解析的 prompt (避免随机性导致编码/取样不一致)
        if getattr(self, "_resolved_prompts", None) is not None:
            prompt = self._resolved_prompts[index]
        else:
            # 用文件 stem 做 deterministic 种子, 保证"两次 __getitem__ 同 index 拿到同 prompt"
            prompt = self._make_prompt(weather, deterministic_seed=str(gt_path))

        if self.defer_transforms:
            # SD3 旧 HF imagefolder 路径 (保留以兼容)
            return {
                "image":              str(gt_path),
                "conditioning_image": str(lq_path),
                "text":               prompt,
                "weather":            weather,
            }

        # eager 模式: 同步返回 tensor, 内部完成预处理.
        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = self.preprocess(gt_img)
        gt_img = self.to_tensor(gt_img)

        lq_img = Image.open(lq_path).convert("RGB")
        lq_img = self.preprocess(lq_img)
        lq_img = self.to_tensor(lq_img)

        result = {
            "conditioning_pixel_values": lq_img,            # LQ, [0, 1]
            "pixel_values":              gt_img * 2.0 - 1.0,  # GT, [-1, 1]
            "weather":                   weather,
        }

        # 优先返回 SD3 预计算的 prompt_embeds / pooled_prompt_embeds
        if getattr(self, "_prompt_embeds", None) is not None:
            result["prompt_embeds"] = self._prompt_embeds[index]
            result["pooled_prompt_embeds"] = self._pooled_prompt_embeds[index]
        else:
            # 兼容旧 SD2 模式: 单 CLIP tokenize
            if self.tokenizer is not None:
                result["input_ids"] = self.tokenize_caption(prompt).squeeze(0)

        return result

    def __len__(self):
        return len(self.samples)


def hf_generator_from_paired(dataset: PairedCaptionDataset):
    """
    把 PairedCaptionDataset(defer_transforms=True) 包装成 HF Dataset.from_generator 可用的 generator.

    字段映射 (yield 文件路径字符串, 调用方按需 .convert("RGB")):
        image_column            (默认 "image")            ← GT 路径
        conditioning_image_column (默认 "conditioning_image") ← LQ 路径
        caption_column          (默认 "text")             ← 已解析 prompt
        weather                                                ← 额外字段, 给评估/分析用

    注意: generator 必须是无状态可重入的, PairedCaptionDataset 自身无状态.
    """
    for idx in range(len(dataset)):
        item = dataset[idx]
        yield {
            "image":              item["image"],
            "conditioning_image": item["conditioning_image"],
            "text":               item["text"],
            "weather":            item["weather"],
        }


def paired_pil_generator(dataset: PairedCaptionDataset):
    """
    Generator for datasets.Dataset.from_generator, 直接 yield PIL Image (RGB).

    列 schema (与 SD3 脚本默认 image_column / conditioning_image_column / caption_column 对齐):
        image              : PIL.Image (RGB GT)
        conditioning_image : PIL.Image (RGB LQ)
        text               : str (已解析的 prompt, weather-aware 随机化已完成)
        weather            : str

    SD3 训练脚本的 preprocess_train 会再做一次 .convert("RGB") + Resize/CenterCrop/ToTensor,
    对已 RGB 的 PIL 是 no-op, 所以这里直接 yield RGB 即可.
    """
    for idx in range(len(dataset)):
        item = dataset[idx]
        yield {
            "image":              Image.open(item["image"]).convert("RGB"),
            "conditioning_image": Image.open(item["conditioning_image"]).convert("RGB"),
            "text":               item["text"],
            "weather":            item["weather"],
        }


def build_paired_hf_dataset(args) -> "datasets.DatasetDict":
    """
    构建 HF DatasetDict (与 SD3 训练脚本的 preprocess_train + with_transform + map 流程兼容).

    数据流:
        1. 扫描 args.dataset_root/{weather}/{split}/{GT,LQ}/
        2. PairedCaptionDataset(defer_transforms=True) 返回路径 + 已解析 prompt
        3. paired_pil_generator 加载 PIL, 通过 datasets.Dataset.from_generator 包装
        4. 返回 DatasetDict({"train": Dataset}) 与 HF imagefolder 流程同 schema

    副作用:
        - 强制 args.proportion_empty_prompts = 0 (让 SD3 的 process_captions 变 passthrough,
          因为天气 prompt 已经在 PairedCaptionDataset._make_prompt 里解析好了)

    Returns:
        datasets.DatasetDict with key "train" -> Dataset columns:
            image (PIL.Image), conditioning_image (PIL.Image), text (str), weather (str)
    """
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

    # 解析 weather_prompts (CLI 的 key:value 列表 → dict)
    weather_prompts_dict = None
    if args.weather_prompts:
        weather_prompts_dict = {}
        for item in args.weather_prompts:
            if ":" in item:
                k, v = item.split(":", 1)
                weather_prompts_dict[k.strip()] = v.strip()

    # 按 weather 限制样本数 (None / <=0 表示不限制)
    weather_num_samples = {}
    for w in args.weather_types:
        attr = f"{w}_num"
        v = getattr(args, attr, None)
        if v is not None and v > 0:
            weather_num_samples[w] = v

    paired = PairedCaptionDataset(
        dataset_root=args.dataset_root,
        weather_types=args.weather_types,
        splits=args.splits,
        tokenizer=None,           # SD3 流程不依赖单 tokenizer, 由 3 编码器在 map() 中处理
        use_prompt=args.use_prompt,
        prompt_ratio=args.prompt_ratio,
        weather_prompts=weather_prompts_dict,
        resolution=args.resolution,
        weather_num_samples=weather_num_samples,
        defer_transforms=True,
    )

    # 强制 process_captions passthrough
    # (避免 SD3 默认 proportion_empty_prompts 在我们的 prompt 上又做一次空串替换)
    args.proportion_empty_prompts = 0

    features = Features({
        "image":              HFImage(),
        "conditioning_image": HFImage(),
        "text":               Value("string"),
        "weather":            Value("string"),
    })

    hf_train = Dataset.from_generator(
        lambda: paired_pil_generator(paired),
        features=features,
    )
    return DatasetDict({"train": hf_train})