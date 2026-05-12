# pip install torch transformers pandas matplotlib

import gc
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# Порог из статьи LLM.int8().
OUTLIER_THRESHOLD = 6.0

# Модели небольшие, чтобы запустилось на обычном компьютере.
MODEL_NAMES = [
    "sshleifer/tiny-gpt2",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "distilgpt2",
]

# Текстов мало, чтобы всё считалось быстрее.
TEXTS = [
    "Large language models can solve different natural language processing tasks.",
    "Quantization reduces memory usage and makes inference faster.",
    "Some transformer activations become unusually large in certain hidden dimensions.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 48


def count_parameters(model):
    # Считаем размер модели.
    return sum(p.numel() for p in model.parameters())


def is_target_module(module):
    # В GPT-2 часть слоёв сделана как Conv1D, поэтому тоже берём.
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


@torch.no_grad()
def measure_outliers(model_name):
    # Загружаем модель.
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    num_params = count_parameters(model)

    total_values = 0
    outlier_values = 0

    total_features = 0
    outlier_features = 0

    feature_masks = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            nonlocal total_values, outlier_values

            x = inputs[0]

            if not torch.is_tensor(x):
                return

            # Берём активации перед линейным слоем.
            x = x.detach()

            # Тут ищем выбросы.
            outlier_mask = x.abs() >= OUTLIER_THRESHOLD

            total_values += x.numel()
            outlier_values += outlier_mask.sum().item()

            # Смотрим, в каких признаках были выбросы.
            channel_mask = outlier_mask.reshape(
                -1,
                outlier_mask.shape[-1]
            ).any(dim=0).cpu()

            if layer_name not in feature_masks:
                feature_masks[layer_name] = torch.zeros_like(
                    channel_mask,
                    dtype=torch.bool
                )

            feature_masks[layer_name] |= channel_mask

        return hook

    # Вешаем hooks на нужные слои.
    for name, module in model.named_modules():
        if is_target_module(module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Прогоняем тексты.
    for text in TEXTS:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        model(**encoded)

    # Убираем hooks.
    for hook in hooks:
        hook.remove()

    # Считаем признаки с выбросами.
    for mask in feature_masks.values():
        total_features += mask.numel()
        outlier_features += mask.sum().item()

    result = {
        "model": model_name,
        "params": num_params,
        "params_millions": num_params / 1_000_000,
        "outlier_values": outlier_values,
        "total_values": total_values,
        "outlier_value_ratio": outlier_values / total_values if total_values else 0,
        "outlier_features": outlier_features,
        "total_features": total_features,
        "outlier_feature_ratio": outlier_features / total_features if total_features else 0,
    }

    # Чистим память.
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    results = []

    # Запускаем модели по очереди.
    for model_name in MODEL_NAMES:
        try:
            result = measure_outliers(model_name)
            results.append(result)

            print(
                f"{result['model']}: "
                f"{result['params_millions']:.2f}M params, "
                f"{result['outlier_values']} outlier values, "
                f"{result['outlier_features']} outlier features"
            )

        except Exception as error:
            # Если одна модель сломалась, остальные всё равно считаются.
            print(f"\nModel {model_name} was skipped because of an error:")
            print(error)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(results) == 0:
        print("No models were processed successfully.")
        return

    # Делаем таблицу.
    df = pd.DataFrame(results)
    df = df.sort_values("params_millions")

    print("\nFinal table:")
    print(df)

    # Сохраняем результаты.
    df.to_csv("outlier_results_light.csv", index=False)

    # Подписи для графиков.
    df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1])
    df["model_label"] = df.apply(
        lambda row: f"{row['model_short']}\n{row['params_millions']:.1f}M",
        axis=1
    )

    # Делаем один общий рисунок.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Первый график.
    bars = axes[0].bar(
        df["model_label"],
        df["outlier_values"],
    )

    axes[0].set_title("Outlier activation values")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y")

    for bar in bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Второй график.
    bars = axes[1].bar(
        df["model_label"],
        df["outlier_features"],
    )

    axes[1].set_title("Outlier features")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y")

    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Третий график.
    bars = axes[2].bar(
        df["model_label"],
        df["outlier_value_ratio"],
    )

    axes[2].set_title("Outlier activation share")
    axes[2].set_xlabel("Model")
    axes[2].set_ylabel("Share")
    axes[2].grid(axis="y")

    for bar in bars:
        height = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"Outlier statistics by model, threshold = {OUTLIER_THRESHOLD}",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig("outlier_statistics_collage.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
