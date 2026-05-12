# pip install torch transformers pandas matplotlib

import gc
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# Берём порог из идеи LLM.int8().
# Если модуль активации >= 6, считаем это выбросом.
OUTLIER_THRESHOLD = 6.0

# Берём 5 небольших моделей, чтобы компьютер не умер.
MODEL_NAMES = [
    "sshleifer/tiny-gpt2",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "distilgpt2",
]

# Маленький набор текстов, чтобы не грузить модель слишком сильно.
TEXTS = [
    "Large language models can solve different natural language processing tasks.",
    "Quantization reduces memory usage and makes inference faster.",
    "Some transformer activations become unusually large in certain hidden dimensions.",
]

# Если есть видеокарта, используем её. Если нет — обычный процессор.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Длина текста маленькая, потому что компьютер не самый мощный.
MAX_LENGTH = 48


def count_parameters(model):
    # Просто считаем все параметры модели.
    return sum(p.numel() for p in model.parameters())


def is_target_module(module):
    # В разных моделях линейные слои называются по-разному.
    # Поэтому учитываем и обычный Linear, и Conv1D из GPT-2.
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


@torch.no_grad()
def measure_outliers(model_name):
    # Загружаем одну модель.
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # У некоторых моделей нет pad token, поэтому ставим eos token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    # Считаем размер модели.
    num_params = count_parameters(model)

    total_values = 0
    outlier_values = 0

    total_features = 0
    outlier_features = 0

    # Тут будем хранить, в каких признаках были выбросы.
    feature_masks = {}

    # Тут будут hooks, чтобы потом их удалить.
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            nonlocal total_values, outlier_values

            x = inputs[0]

            if not torch.is_tensor(x):
                return

            # x — это вход в линейный слой.
            # Обычно форма такая: batch, длина текста, hidden size.
            x = x.detach()

            abs_x = x.abs()

            # Тут проверяем, какие значения стали выбросами.
            outlier_mask = abs_x >= OUTLIER_THRESHOLD

            total_values += x.numel()
            outlier_values += outlier_mask.sum().item()

            # Признак считается выбросным, если хотя бы один раз там был выброс.
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

    # Вешаем hooks на линейные слои.
    for name, module in model.named_modules():
        if is_target_module(module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Прогоняем тексты через модель.
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

    # Hooks больше не нужны.
    for hook in hooks:
        hook.remove()

    # Считаем, сколько признаков были выбросными.
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

    # Выгружаем модель, чтобы не забивать память.
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    results = []

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
            # Если одна модель не загрузилась, весь код не падает.
            print(f"\nModel {model_name} was skipped because of an error:")
            print(error)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(results) == 0:
        print("No models were processed successfully.")
        return

    # Делаем таблицу с результатами.
    df = pd.DataFrame(results)
    df = df.sort_values("params_millions")

    print("\nFinal table:")
    print(df)

    # Сохраняем таблицу, чтобы потом можно было вставить в отчёт.
    df.to_csv("outlier_results_light.csv", index=False)

    # Делаем короткие названия моделей для графиков.
    df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1])

    # В подпись добавляем размер модели.
    df["model_label"] = df.apply(
        lambda row: f"{row['model_short']}\n{row['params_millions']:.1f}M",
        axis=1
    )

    # Первый график: сколько всего значений-выбросов.
    plt.figure(figsize=(10, 5))

    bars = plt.bar(
        df["model_label"],
        df["outlier_values"],
    )

    plt.xlabel("Model")
    plt.ylabel("Number of outlier activation values")
    plt.title(f"Outlier activation values by model, threshold = {OUTLIER_THRESHOLD}")
    plt.grid(axis="y")
    plt.tight_layout()

    # Подписываем числа над столбиками.
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.savefig("outlier_values_bar.png", dpi=200)
    plt.show()

    # Второй график: в скольких признаках были выбросы.
    plt.figure(figsize=(10, 5))

    bars = plt.bar(
        df["model_label"],
        df["outlier_features"],
    )

    plt.xlabel("Model")
    plt.ylabel("Number of outlier features")
    plt.title(f"Outlier features by model, threshold = {OUTLIER_THRESHOLD}")
    plt.grid(axis="y")
    plt.tight_layout()

    # Опять подписываем числа над столбиками.
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.savefig("outlier_features_bar.png", dpi=200)
    plt.show()

    # Третий график: доля выбросов среди всех значений.
    plt.figure(figsize=(10, 5))

    bars = plt.bar(
        df["model_label"],
        df["outlier_value_ratio"],
    )

    plt.xlabel("Model")
    plt.ylabel("Share of outlier activation values")
    plt.title(f"Outlier activation share by model, threshold = {OUTLIER_THRESHOLD}")
    plt.grid(axis="y")
    plt.tight_layout()

    # Тут числа маленькие, поэтому выводим 6 знаков после запятой.
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.savefig("outlier_value_ratio_bar.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()