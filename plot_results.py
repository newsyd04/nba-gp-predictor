import os
import matplotlib.pyplot as plt

def plot_average_and_max_fitness_history(average_fitness_history: list, max_fitness_history: list) -> None:
    os.makedirs("results", exist_ok=True)

    plt.plot(average_fitness_history, marker='o', label='Average Fitness')
    plt.plot(max_fitness_history, marker='x', label='Max Fitness')
    plt.title("Average and Max Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/average_and_max_fitness_history.png")
    plt.show()
    plt.close()

def plot_roc_auc_for_test_results(predictions: list[float], targets: list[int]) -> None:
    """
    Plots the ROC (Receiver Operating Characteristic) curve and calculates the AUC (Area Under Curve) for test results.
    """
    os.makedirs("results", exist_ok=True)

    thresholds = sorted(set(predictions))
    true_positive_rate_list = []
    false_positive_rate_list = []

    actual_positives = sum(targets)
    actual_negatives = len(targets) - actual_positives

    for thresh in thresholds:
        true_positive = false_positive = true_negative = false_negative = 0
        for prediction, actual in zip(predictions, targets):
            pred_label = 1 if prediction >= thresh else 0

            if pred_label == 1 and actual == 1:
                true_positive += 1
            elif pred_label == 1 and actual == 0:
                false_positive += 1
            elif pred_label == 0 and actual == 0:
                true_negative += 1
            elif pred_label == 0 and actual == 1:
                false_negative += 1

        true_positive_rate = true_positive / actual_positives if actual_positives > 0 else 0
        false_positive_rate = false_positive / actual_negatives if actual_negatives > 0 else 0

        true_positive_rate_list.append(true_positive_rate)
        false_positive_rate_list.append(false_positive_rate)

    roc_points = sorted(zip(false_positive_rate_list, true_positive_rate_list))
    false_positive_rate_list, true_positive_rate_list = zip(*roc_points)

    # area under curve using trapezoidal rule
    area_under_curve = 0.0
    for i in range(1, len(false_positive_rate_list)):
        area_under_curve += (false_positive_rate_list[i] - false_positive_rate_list[i-1]) * (true_positive_rate_list[i] + true_positive_rate_list[i-1]) / 2
    plt.plot(false_positive_rate_list, true_positive_rate_list, label=f"ROC Curve (AUC = {area_under_curve:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set Performance")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/test_roc_curve.png")
    plt.show()
    plt.close()

    print(f"Test ROC AUC: {area_under_curve:.6f}")
