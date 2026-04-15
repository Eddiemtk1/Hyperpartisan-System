#pip install pandas scikit-learn requests
#pip install matplotlib

import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# 1. Configuration
API_URL = "http://127.0.0.1:8000/analyse"
CSV_FILE = "semeval_random_sample_with_titles[20].csv"


def run_evaluation():
    print("Loading dataset...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILE}. Please create it first.")
        return
    # 1. Separate the dataframe into hyperpartisan and neutral
    df_true = df[df["true_label"] == True]
    df_false = df[df["true_label"] == False]

    # 2. Randomly sample 20 from each
    # (Remove random_state=42 if you want different articles every single time you run the script)
    sampled_true = df_true.sample(n=20, random_state=42)
    sampled_false = df_false.sample(n=20, random_state=42)

    # 3. Combine them and shuffle the final list
    df = (
        pd.concat([sampled_true, sampled_false])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    actual_labels = []
    predicted_labels = []

    print(
        f"Starting evaluation of {len(df)} articles. This may take a few minutes...\n"
    )

    # 2. Iterate through the dataset
    for index, row in df.iterrows():
        # --- NEW CODE: Extract and combine Title and Text ---
        article_title = str(row["title"])
        article_body = str(row["text"])

        # Combine them with a couple of newlines so the LLM reads it naturally
        combined_text = f"{article_title}\n\n{article_body}"

        true_label = bool(row["true_label"])

        # Send the COMBINED text to your local FastAPI server
        try:
            response = requests.post(API_URL, json={"text": combined_text})

            if response.status_code == 200:
                result = response.json()
                predicted_label = result["is_hyperpartisan"]

                actual_labels.append(true_label)
                predicted_labels.append(predicted_label)

                # Print the title in the console so you can see which article is being processed!
                short_title = (
                    article_title[:40] + "..."
                    if len(article_title) > 40
                    else article_title
                )
                print(
                    f"[{index + 1}/{len(df)}] {short_title} | Actual={true_label} -> Predicted={predicted_label}"
                )
            else:
                print(f"[{index + 1}/{len(df)}] API Error {response.status_code}")

        except Exception as e:
            print(f"[{index + 1}/{len(df)}] Connection Error (Is Uvicorn running?)")

    # 3. Calculate and Print Metrics
    print("\n" + "=" * 40)
    print("🏆 EVALUATION RESULTS 🏆")
    print("=" * 40)

    # Calculate base metrics
    acc = accuracy_score(actual_labels, predicted_labels)
    prec = precision_score(actual_labels, predicted_labels)
    rec = recall_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)

    print(f"Accuracy:  {acc:.2f} (Overall correctness)")
    print(f"Precision: {prec:.2f} (When it flags bias, how often is it right?)")
    print(f"Recall:    {rec:.2f} (How much of the total bias did it catch?)")
    print(f"F1-Score:  {f1:.2f} (Balance between Precision and Recall)")

    print("\n--- Confusion Matrix ---")
    tn, fp, fn, tp = confusion_matrix(actual_labels, predicted_labels).ravel()
    print(f"True Positives (Correctly flagged as biased): {tp}")
    print(f"True Negatives (Correctly flagged as safe):   {tn}")
    print(f"False Positives (Falsely flagged as biased):  {fp}")
    print(f"False Negatives (Missed the bias completely): {fn}")

    print("\n--- Detailed Report ---")
    print(
        classification_report(
            actual_labels, predicted_labels, target_names=["Neutral", "Hyperpartisan"]
        )
    )

# --- NEW CODE: Visual Confusion Matrix Plot ---
    print("\nGenerating Confusion Matrix Plot...")
    cm = confusion_matrix(actual_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Hyperpartisan"])
    
    # Plot it with a blue color map
    disp.plot(cmap=plt.cm.Blues)
    plt.title("TruthLens Evaluation: Confusion Matrix")
    
    # Show the plot in a pop-up window
    plt.show() 

    # Tip: If you want to save it as an image instead of having a pop-up, 
    # comment out plt.show() and uncomment the line below:
    # plt.savefig('confusion_matrix.png', bbox_inches='tight')

if __name__ == "__main__":
    run_evaluation()
