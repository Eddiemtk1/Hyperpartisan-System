#pip install pandas scikit-learn requests
#pip install matplotlib

import requests
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    matthews_corrcoef
)

# 1. Configuration
API_URL = "http://127.0.0.1:8000/analyse"
CSV_FILE = "semeval_random_sample_with_titles[50].csv"

def run_evaluation():
    print("Loading dataset...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILE}. Please create it first.")
        return

    # Trackers for results and speed
    actual_labels = []
    predicted_labels = []
    latencies = []       
    detailed_data = []   

    print(f"Starting evaluation of {len(df)} articles. This may take a few minutes...\n")

    # 2. Iterate through the dataset
    for index, row in df.iterrows():
        article_title = str(row["title"])
        article_body = str(row["text"])
        combined_text = f"{article_title}\n\n{article_body}"
        true_label = bool(row["true_label"])

        try:
            start_time = time.perf_counter()
            
            response = requests.post(API_URL, json={"text": combined_text})
            
            end_time = time.perf_counter()

            if response.status_code == 200:
                elapsed_time = end_time - start_time
                latencies.append(elapsed_time)
                
                result = response.json()
                predicted_label = result["is_hyperpartisan"]

                actual_labels.append(true_label)
                predicted_labels.append(predicted_label)

                detailed_data.append({
                    "Article_Number": index + 1,
                    "Title": article_title,
                    "Latency_Seconds": round(elapsed_time, 3)
                })

                short_title = article_title[:40] + "..." if len(article_title) > 40 else article_title
                print(f"[{index + 1}/{len(df)}] {short_title} | Actual={true_label} -> Predicted={predicted_label} | Time={elapsed_time:.2f}s")
            else:
                print(f"[{index + 1}/{len(df)}] API Error {response.status_code}")

        except Exception as e:
            print(f"[{index + 1}/{len(df)}] Connection Error (Is Uvicorn running?)")

    #Export the Raw Data to a new CSV
    if detailed_data:
        results_df = pd.DataFrame(detailed_data)
        results_df.to_csv("evaluation_latencies.csv", index=False)
        print("\n💾 Saved raw latency data to 'evaluation_latencies.csv'")

#Generate the Latency Distribution Graph
    if latencies:
        import seaborn as sns # Ensure this is imported
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Create a histogram to show distribution
        sns.histplot(latencies, bins=10, kde=True, color="royalblue", edgecolor="black")
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        plt.axvline(mean_latency, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_latency:.2f}s')
        plt.axvline(p95_latency, color='orange', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95_latency:.2f}s')
        
        plt.title('TruthLens Latency Distribution (Zero-Shot Inference)', fontsize=14, fontweight='bold')
        plt.xlabel('Latency (Seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        
        plt.savefig("latency_histogram1.png", dpi=300, bbox_inches='tight')
        print("✅ Saved Latency Histogram to 'latency_histogram1.png'")
        plt.clf() # Clear the figure

    # Calculate and print the metrics
    print("\n" + "=" * 40)
    print("🏆 EVALUATION RESULTS 🏆")
    print("=" * 40)

    if actual_labels:
        acc = accuracy_score(actual_labels, predicted_labels)
        prec = precision_score(actual_labels, predicted_labels)
        rec = recall_score(actual_labels, predicted_labels)
        f1 = f1_score(actual_labels, predicted_labels)
        mcc = matthews_corrcoef(actual_labels, predicted_labels)

        print(f"Accuracy:  {acc:.2f} (Overall correctness)")
        print(f"Precision: {prec:.2f} (When it flags bias, how often is it right?)")
        print(f"Recall:    {rec:.2f} (How much of the total bias did it catch?)")
        print(f"F1-Score:  {f1:.2f} (Balance between Precision and Recall)")
        print(f"MCC:       {mcc:.2f} (Overall quality score from -1 to +1)")

        print("\n--- Speed Metrics ---")
        times_array = np.array(latencies)
        print(f"Average (Mean):  {np.mean(times_array):.2f} seconds")
        print(f"95th Percentile: {np.percentile(times_array, 95):.2f} seconds")

        print("\n--- Detailed Report ---")
        print(classification_report(actual_labels, predicted_labels, target_names=["Neutral", "Hyperpartisan"]))

        # Generate the Confusion Matrix Visual
        print("\nGenerating Confusion Matrix Plot...")
        cm = confusion_matrix(actual_labels, predicted_labels)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Hyperpartisan"])
        
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        
        plt.title("TruthLens Evaluation: Confusion Matrix", fontsize=14)
        ax.set_ylabel("Actual label", fontsize=12) 
        ax.set_xlabel("Predicted label", fontsize=12)

        ax.grid(False)
        
        #Save it
        plt.tight_layout()
        plt.savefig('confusion_matrix1.png', dpi=300, bbox_inches='tight')
        print("🟦 Saved Confusion Matrix to 'confusion_matrix1.png'")

if __name__ == "__main__":
    run_evaluation()