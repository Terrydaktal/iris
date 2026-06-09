from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


class TextTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(input_ids)
        if features.ndim > 2:
            features = features.flatten(2).mean(dim=-1)
        return F.normalize(features.float(), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-hub:timm/ViT-L-16-SigLIP2-384")
    parser.add_argument("--out-dir", type=Path, default=Path("models/clip-text"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import open_clip
    from transformers import AutoTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model.removeprefix("hf-hub:"), use_fast=True)
    tokenizer.save_pretrained(args.out_dir)

    model, _, _ = open_clip.create_model_and_transforms(args.model, device="cpu")
    model.eval()
    wrapper = TextTower(model).eval()
    dummy = open_clip.get_tokenizer(args.model)(["test phrase"])
    if dummy.shape[1] != 64:
        raise RuntimeError(f"expected context length 64, got {tuple(dummy.shape)}")

    output_path = args.out_dir / "clip_text.onnx"
    torch.onnx.export(
        wrapper,
        (dummy.to(torch.long),),
        output_path,
        input_names=["input_ids"],
        output_names=["text_features"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_features": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=True,
    )
    print(f"wrote {output_path}")
    print(f"wrote {args.out_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
