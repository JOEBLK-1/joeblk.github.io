from pathlib import Path
import csv

def load_labels(path, is_csv=False):
    labels = {}
    with path.open("r", encoding="utf-8", newline="" if is_csv else None) as f:
        reader = csv.reader(f) if is_csv else (line.split() for line in f)
        for row in reader:
            if len(row) < 2: continue
            labels[Path(row[0]).name] = int(row[1])
    return labels

def compute_accuracy(ground_truth, predictions):
    matches = sum(1 for k, v in ground_truth.items() if predictions.get(k) == v)
    total = len(ground_truth)
    accuracy = matches / total if total else 0
    print(f"准确率: {accuracy:.2%} ({matches}/{total})")
    print("\n预测不匹配:")
    for k, v in ground_truth.items():
        if k in predictions and predictions[k] != v:
            print(f"  {k}: expected {v}, got {predictions[k]}")

def main():
    ground_truth = load_labels(Path("test_list 篡改版.txt"))
    predictions = load_labels(Path("csv_file_path.csv"), is_csv=True)
    compute_accuracy(ground_truth, predictions)

if __name__ == "__main__":
    main()
