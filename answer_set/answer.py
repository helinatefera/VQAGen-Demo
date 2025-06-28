import pandas as pd

# Load your CSV
df = pd.read_csv("/home/amerti/Documents/MSC/1demonstration/answer_set/answer.csv")  # replace with your actual CSV file path

# Combine predicted and actual answers
all_answers = pd.concat([df["predicted_answer"], df["actual_answer"]]).dropna().unique()

# Build index mapping
answer_dict = {i: f'"{ans}"' for i, ans in enumerate(sorted(set(all_answers)))}

# Save to txt
with open("am_answer_set.txt", "w", encoding="utf-8") as f:
    for idx, ans in answer_dict.items():
        f.write(f"{idx}: {ans}\n")
