import os
import time
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import joblib

from mealpy import FloatVar
from mealpy.evolutionary_based import GA

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_flatten_images(real_dir, fake_dir, img_size=(224, 224), max_per_class=5000):
    X, y = [], []

    for label, folder in [(0, real_dir), (1, fake_dir)]:
        files = sorted(os.listdir(folder))[:max_per_class]

        for fname in tqdm(files, desc=f"Loading {os.path.basename(folder)}"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, fname)

            try:
                img = Image.open(path).convert("RGB").resize(img_size)
                img_arr = np.array(img).transpose(2, 0, 1).flatten()

                X.append(img_arr)
                y.append(label)

            except Exception as e:
                print(f"Error loading {path}: {e}")

    return np.array(X), np.array(y)


def apply_anova_selection(X, y, k=5000, output_dir="outputs/feature_selection"):
    os.makedirs(output_dir, exist_ok=True)

    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selector_path = os.path.join(output_dir, "selector_kbest.pkl")
    joblib.dump(selector, selector_path)

    selected_indices = selector.get_support(indices=True)
    np.savetxt(
        os.path.join(output_dir, "anova_selected_indices.csv"),
        selected_indices,
        fmt="%d",
        delimiter=","
    )

    return X_selected, selector, selected_indices


def run_ga_selection(
    X,
    y,
    epoch=30,
    pop_size=20,
    pc=0.9,
    pm=0.05,
    output_dir="outputs/feature_selection",
):
    os.makedirs(output_dir, exist_ok=True)

    def fitness_func(solution):
        mask = np.array(solution) > 0.5

        if np.sum(mask) == 0:
            return 1.0

        X_selected = X[:, mask]

        X_train, X_val, y_train, y_val = train_test_split(
            X_selected,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = accuracy_score(y_val, preds)

        return 1.0 - acc

    num_features = X.shape[1]

    problem = {
        "bounds": FloatVar(
            lb=(0.0,) * num_features,
            ub=(1.0,) * num_features,
            name="feature_selector"
        ),
        "obj_func": fitness_func,
        "minmax": "min",
    }

    model = GA.BaseGA(
        epoch=epoch,
        pop_size=pop_size,
        pc=pc,
        pm=pm
    )

    start_time = time.time()
    best_solution = model.solve(problem, mode="thread")
    runtime = time.time() - start_time

    final_mask = np.array(best_solution.solution) > 0.5
    selected_indices = np.where(final_mask)[0]
    best_accuracy = 1.0 - best_solution.target.fitness

    np.save(os.path.join(output_dir, "ga_best_feature_mask.npy"), final_mask)
    np.savetxt(
        os.path.join(output_dir, "ga_selected_indices.csv"),
        selected_indices,
        fmt="%d",
        delimiter=","
    )

    result = {
        "best_fitness": float(best_solution.target.fitness),
        "best_accuracy": float(best_accuracy),
        "num_selected_features": int(final_mask.sum()),
        "runtime_seconds": float(runtime),
        "epoch": epoch,
        "pop_size": pop_size,
        "pc": pc,
        "pm": pm,
    }

    with open(os.path.join(output_dir, "ga_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    save_ga_history(model, output_dir)

    return model, best_solution, final_mask, result


def save_ga_history(model, output_dir):
    history_dir = os.path.join(output_dir, "history_logs")
    os.makedirs(history_dir, exist_ok=True)

    history_items = {
        "epoch_time.csv": model.history.list_epoch_time,
        "global_best_fit.csv": model.history.list_global_best_fit,
        "current_best_fit.csv": model.history.list_current_best_fit,
        "diversity.csv": model.history.list_diversity,
        "exploration.csv": model.history.list_exploration,
        "exploitation.csv": model.history.list_exploitation,
    }

    for filename, data in history_items.items():
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(history_dir, filename), index=False)


def save_ga_plots(model, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    model.history.save_global_objectives_chart(
        filename=os.path.join(plot_dir, "global_objectives.png")
    )
    model.history.save_local_objectives_chart(
        filename=os.path.join(plot_dir, "local_objectives.png")
    )
    model.history.save_global_best_fitness_chart(
        filename=os.path.join(plot_dir, "global_best_fitness.png")
    )
    model.history.save_local_best_fitness_chart(
        filename=os.path.join(plot_dir, "local_best_fitness.png")
    )
    model.history.save_runtime_chart(
        filename=os.path.join(plot_dir, "runtime_per_epoch.png")
    )
    model.history.save_exploration_exploitation_chart(
        filename=os.path.join(plot_dir, "explore_vs_exploit.png")
    )
    model.history.save_diversity_chart(
        filename=os.path.join(plot_dir, "diversity_chart.png")
    )

    try:
        model.history.save_trajectory_chart(
            list_agent_idx=[0, 1, 2],
            selected_dimensions=[0, 1],
            filename=os.path.join(plot_dir, "trajectory_chart.png")
        )
    except ValueError as e:
        print(f"Skipping trajectory chart: {e}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and flattening images...")
    X, y = load_flatten_images(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        img_size=(args.img_size, args.img_size),
        max_per_class=args.max_per_class
    )

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")

    print("Running ANOVA SelectKBest...")
    X_anova, selector, anova_indices = apply_anova_selection(
        X,
        y,
        k=args.k_best,
        output_dir=args.output_dir
    )

    print(f"ANOVA selected shape: {X_anova.shape}")

    print("Running Genetic Algorithm feature selection...")
    model, best_solution, final_mask, result = run_ga_selection(
        X_anova,
        y,
        epoch=args.epoch,
        pop_size=args.pop_size,
        pc=args.pc,
        pm=args.pm,
        output_dir=args.output_dir
    )

    print(f"Best Accuracy: {result['best_accuracy']:.4f}")
    print(f"Selected Features: {result['num_selected_features']}")

    if args.save_plots:
        print("Saving optimization plots...")
        save_ga_plots(model, args.output_dir)

    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/feature_selection")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_per_class", type=int, default=5000)
    parser.add_argument("--k_best", type=int, default=5000)

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--pc", type=float, default=0.9)
    parser.add_argument("--pm", type=float, default=0.05)

    parser.add_argument("--save_plots", action="store_true")

    args = parser.parse_args()
    main(args)