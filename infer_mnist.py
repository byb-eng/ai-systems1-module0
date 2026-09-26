import os
from pathlib import Path


# 使用脚本所在目录，避免从其他位置运行时改变下载路径。
ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")

import torch
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, ResNetForImageClassification


def main():
    model_name = "microsoft/resnet-18"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备：{device}", flush=True)

    # 加载模型及其配套的图片预处理配置。
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        use_fast=False,
    )
    model = ResNetForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # 只使用全部测试集，不训练或微调模型。
    dataset = MNIST(root=str(ROOT / "data"), train=False, download=True)
    batch_size = 32
    correct = 0
    total = 0

    print(f"模型：{model_name}")
    print(f"模型输出类别数：{model.config.num_labels}")
    print(f"测试图片数：{len(dataset)}")
    print("注意：ImageNet 类别与 MNIST 数字含义不同，下面仅统计编号匹配率。")

    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[i]
                for i in range(start, min(start + batch_size, len(dataset)))
            ]

            # 将单通道灰度图片转换为模型所需的三通道图片。
            images = [image.convert("RGB") for image, _ in samples]
            labels = torch.tensor([label for _, label in samples])

            # 按预训练配置执行 resize、中心裁剪和归一化。
            inputs = processor(
                images=images,
                do_resize=True,
                return_tensors="pt",
            )
            inputs = {
                name: tensor.to(device)
                for name, tensor in inputs.items()
            }

            if start == 0:
                print(f"预处理后的输入形状：{inputs['pixel_values'].shape}")

            logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1).cpu()

            # 直接比较编号仅用于作业演示，不代表有效的数字分类准确率。
            correct += (predictions == labels).sum().item()
            total += len(labels)

            if total % 1024 == 0 or total == len(dataset):
                print(f"已处理：{total}/{len(dataset)}", flush=True)

    accuracy = correct / total
    print(f"\n编号直接比较的准确率：{accuracy:.4%}（{correct}/{total}）")
    print("该指标仅为编号匹配率，不能作为 MNIST 数字识别能力的评价。")


if __name__ == "__main__":
    main()
