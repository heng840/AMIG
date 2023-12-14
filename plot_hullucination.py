import argparse
import os

from matplotlib import pyplot as plt

from cal_hallucination import compute_metrics, aggregate_results
import seaborn as sns
import pandas as pd

from matplotlib.font_manager import FontProperties


def prepare_data_for_plotting(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Iterate through the rows and create a new DataFrame with the required format
    plot_data = []
    for _, row in df.iterrows():
        relation = row["Relation"]
        f1_score = row["F1 Score"]
        folder = row["Folder"]

        # Determine the main title and subtitle based on the folder value
        if "mbert" in folder and "en" in folder:
            main_title = "m-BERT+English"
        elif "mbert" in folder and "zh" in folder:
            main_title = "m-BERT+Chinese"
        elif "mgpt" in folder and "en" in folder:
            main_title = "m-GPT+English"
        elif "mgpt" in folder and "zh" in folder:
            main_title = "m-GPT+Chinese"
        elif "gpt" in folder:
            main_title = "GPT-2+English"
        elif "bert" in folder:
            main_title = "BERT+English"
        else:
            continue  # Skip if folder does not match any known pattern

        subtitle = "wo_DN" if "use_PLMs" in folder else "with_DN"

        plot_data.append({
            "Relation": relation,
            "F1 Score": f1_score,
            "Main Title": main_title,  # Added "Main Title" column
            "Subtitle": subtitle  # Added "Subtitle" column
        })

    return pd.DataFrame(plot_data)

from matplotlib.font_manager import FontProperties

def plot_f1_scores_for_case(metrics_df, case_title, font_path, out_path):
    # Filter the DataFrame to include only the specific case
    metrics_df_case = metrics_df[metrics_df['Main Title'] == case_title]

    # Create a FontProperties object with the font file
    font_properties = FontProperties(fname=font_path)

    sns.set_theme(style="whitegrid")

    # Drawing a barplot with relation as the x-axis and F1 Score as the y-axis
    g = sns.catplot(
        data=metrics_df_case,
        kind="bar",
        x="Relation",
        y="F1 Score",
        hue="Subtitle", # Differentiating between "with_DN" and "wo_DN"
        palette="dark",
        alpha=0.6,
        height=6,
        aspect=4,
    )

    # Customizing the tick labels font properties
    tick_label_font_size = 20  # Adjust this value for both x-axis and y-axis labels

    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties, rotation=60, fontsize=30) # Modified here for x-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties, fontsize=30) # Modified here for y-axis labels
        ax.set_xlabel("Relation name", fontproperties=font_properties,
                      fontsize=36)
        ax.set_ylabel("F1 Score", fontproperties=font_properties,
                      fontsize=36)
    g.legend.set_title(case_title)
    legend_title = g.legend.get_title()
    legend_title.set_fontproperties(font_properties)
    legend_title.set_fontsize(36)
    g.legend.set_bbox_to_anchor((1.155, 0.5))
    # Modify the font size of specific legend labels

    plt.yticks(ticks=g.ax.get_yticks(), labels=[f'{val:.2f}' for val in g.ax.get_yticks()])
    for text in g.legend.texts:
        text.set_font_properties(font_properties)
        text.set_fontsize(36)

    plt.tight_layout()
    # Saving the plot
    g.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Arguments for pararel result plotting')
    parser.add_argument('--csv_file_path',
                        # default='723final/combined_metrics.csv',
                        default='723/combined_metrics.csv',
                        type=str,)
    args = parser.parse_args()
    # Path to the CSV file
    csv_file_path = args.csv_file_path

    # Path to the font file
    font_path = 'times new roman.ttf'

    # Prepare data for plotting
    plot_data_df = prepare_data_for_plotting(csv_file_path)

    cases = [
        # "m-BERT+English",
        # "m-BERT+Chinese",
        # "m-GPT+English",
        # "m-GPT+Chinese",
        'BERT+English',
        'GPT-2+English',
    ]

    # Plot the F1 Scores for each case
    os.makedirs('h_appendix', exist_ok=True)
    for case in cases:
        out_path = f'h_appendix/f1_scores_{case.replace("+", "_").replace(" ", "_")}.png'
        plot_f1_scores_for_case(plot_data_df, case, font_path, out_path)

