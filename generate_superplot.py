
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind, levene
import pandas as pd
from tqdm import tqdm

sns.set(style="whitegrid")

def calculate_mean_per_frame(tif_path):
    img = tifffile.imread(tif_path)
    img = np.squeeze(img)
    means = []
    for z in range(img.shape[0]):
        frame = img[z].flatten()
        frame = frame[frame > 0]  # Exclude zeros
        if len(frame) > 0:
            means.append(np.mean(frame))
        else:
            means.append(0)  # or np.nan if you prefer to skip
    return means

def process_folder(folder_path):
    all_means = []
    filenames = []
    for filename in tqdm(sorted(os.listdir(folder_path)), desc="Processing files"):
        if filename.lower().endswith(".tif"):
            full_path = os.path.join(folder_path, filename)
            means = calculate_mean_per_frame(full_path)
            all_means.append(means)
            filenames.append(filename)

    return all_means, filenames

def generate_superplot(control_folder, mutation_folder, out_path):
    control_data,  control_filenames = process_folder(control_folder)
    mutation_data, mutation_filenames = process_folder(mutation_folder)

    data = []
    for i, means in enumerate(control_data):
        for value in means:
            data.append({"Condition": "Control", "Replicate": i, "Value": value})
    for i, means in enumerate(mutation_data):
        for value in means:
            data.append({"Condition": "CPF_CYP", "Replicate": i, "Value": value})
    df = pd.DataFrame(data)

    control_means = [np.mean(x) for x in control_data]
    mutation_means = [np.mean(x) for x in mutation_data]

    var_control  = [np.var(x, ddof=1) for x in control_data] 
    var_mutation = [np.var(x, ddof=1) for x in mutation_data]

    stat, p_var = levene(control_means, mutation_means)
    print(f"Levene test p = {p_var:.4f}")

    #t_stat, p_value = ttest_rel(control_means, mutation_means)
    t_stat, p_value = ttest_ind(control_means, mutation_means, equal_var=(p_var >= 0.05))


    for i, means in enumerate(control_means):
        print(control_filenames[i], ": ", means)

    for i, means in enumerate(mutation_means):
        print(mutation_filenames[i], ": ", means)


    # Build rows
    control_df = pd.DataFrame({
        "Replicate": [f"R{i+1}" for i in range(len(control_means))],
        "Condition": "Control",
        "Value": control_means
    })

    mutation_df = pd.DataFrame({
        "Replicate": [f"R{i+1}" for i in range(len(mutation_means))],
        "Condition": "CPF_CYP",  # or "Drug" depending on your naming
        "Value": mutation_means
    })

    # Combine into one DataFrame
    df_means = pd.concat([control_df, mutation_df], ignore_index=True)

    plt.figure(figsize=(5, 7))
    ax = sns.stripplot(data=df, x="Condition", y="Value", color="gray", jitter=0.2, alpha=0.3, size=1)

 
    # Box plot instead of mean  std error bars
    ax = sns.boxplot(data=df_means, x="Condition", y="Value", showcaps=True, boxprops=dict(alpha=0.5))
    ax = sns.stripplot(data=df_means, x="Condition", y="Value", jitter=0.15, alpha=1.0, size=8)


    ax.set_xlim(-0.50, 1.5)
    ax.set_ylim(0, 1)
    plt.text(0.5, 0.9, f"P = {p_value:.3f}", ha='center')
    plt.ylabel("Order Parameter [S]")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()

if __name__ == "__main__":

    control_folder = "/medicina/hmorales/projects/AnemoneTracking/data/processed/Control_orderparam_flow_smooth/"
    mutation_folder = "/medicina/hmorales/projects/AnemoneTracking/data/processed/CPF_CYP_orderparam_flow_smooth/"
    out_path = "/medicina/hmorales/projects/AnemoneTracking/data/processed/orderparam_flow_smooth.pdf"
    generate_superplot(control_folder, mutation_folder, out_path)
