import matplotlib.pyplot as plt
import numpy as np
from captum.concept._utils.common import concepts_to_str
from scipy import stats
import matplotlib.pyplot as plt



def format_float(f):
    return float("{:.3f}".format(f) if abs(f) >= 0.0005 else "{:.3e}".format(f))


def plot_tcav_scores(experimental_sets, tcav_scores, layer_name_list):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layer_name_list))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i - 1]])
        _ax = ax[idx_es] if len(experimental_sets) > 1 else ax
        for i in range(len(concepts)):
            val = [
                format_float(scores["sign_count"][i])
                for layer, scores in tcav_scores[concepts_key].items()
            ]
            direction = [
                format_float(scores["magnitude"][i])
                for layer, scores in tcav_scores[concepts_key].items()
            ]
            print(direction)
            _ax.bar(
                pos[i], val, width=barWidth, edgecolor="white", label=concepts[i].name
            )

        # Add xticks on the middle of the group bars
        _ax.set_xlabel("Set {}".format(str(idx_es)), fontweight="bold", fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layer_name_list))])
        _ax.set_xticklabels(layer_name_list, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(
            fontsize=16, bbox_to_anchor=(1.3, 1), loc="upper right"
        )  # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


# Function to calculate mean, confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    mean, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, h


def get_confidnece_plot(
    mytcav,
    exp_set,
    target_layer_name,
    score_type,
    target_tensor,
    device,
    alpha=0.05,
    label_name=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    if label_name is None:
        label_name = "target"

    scores = mytcav.interpret(
        inputs=target_tensor.to(device), experimental_sets=exp_set, target=1, n_steps=5
    )

    target_score_list = list()
    random_score_list = list()
    for key, value in scores.items():
        trial = scores[key]
        if len(trial) == 0:
            continue
        target_score, random_score = trial[target_layer_name][score_type][:]
        target_score_list.append(target_score.cpu().item())
        random_score_list.append(random_score.cpu().item())

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(target_score_list, random_score_list)

    if p_value < alpha:  # alpha value is 0.05 or 5%
        relation = "Disjoint"
    else:
        relation = "Overlap"

    # Calculate means and confidence intervals
    mean1, h1 = mean_confidence_interval(target_score_list)
    mean2, h2 = mean_confidence_interval(random_score_list)

    # Plotting
    fig = plt.figure(figsize=(8, 6))

    # Bar plot with error bars
    plt.bar(
        [f"{label_name}", "random"],
        [mean1, mean2],
        yerr=[h1, h2],
        color=["blue", "green"],
        capsize=10,
    )
    plt.ylim(bottom=-0.1)
    plt.ylabel("Mean score")
    plt.title(
        "Comparison of Two concept with Confidence Intervals:{}  \n p-value = {:.4f}".format(
            relation, p_value
        )
    )

    plt.show()
    return fig, (mean1, h1), (mean2, h2), p_value, random_score_list


def draw_heatmap(
    matrix,
    row_names=None,
    col_names=None,
    cmap="bwr",
    cell_width=1,
    cell_height=1,
    vmin=None,
    vmax=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Draw a heatmap for a given matrix using the specified colormap.
    
    Parameters:
    - matrix (list of lists or numpy array): The input N x M matrix.
    - row_names (list of str, optional): Names of rows.
    - col_names (list of str, optional): Names of columns.
    - cmap (str, optional): The colormap to use. Default is 'bwr' (blue-white-red).
    - cell_width (float, optional): Width of each cell in the heatmap. Default is 1.
    - cell_height (float, optional): Height of each cell in the heatmap. Default is 1.
    - vmin (float, optional): Minimum value for colormap scaling.
    - vmax (float, optional): Maximum value for colormap scaling.
    
    Returns:
    - None
    """
    fig_width = len(matrix[0]) * cell_width
    fig_height = len(matrix) * cell_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cax = ax.matshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Display the intensity values in each cell
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    # Set row and column names
    if row_names:
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names)
    if col_names:
        ax.set_xticks(np.arange(len(col_names)))
        ax.set_xticklabels(col_names, rotation=45, ha="right")
        ax.xaxis.set_ticks_position("bottom")

    plt.colorbar(cax)
    plt.show()
    return fig


def draw_heatmap(matrix,ci_matrix=None, row_names=None, col_names=None, cmap='bwr', cell_width=1, cell_height=1, vmin=None, vmax=None):
    """
    Draw a heatmap for a given matrix using the specified colormap.
    
    Parameters:
    - matrix (list of lists or numpy array): The input N x M matrix.
    - row_names (list of str, optional): Names of rows.
    - col_names (list of str, optional): Names of columns.
    - cmap (str, optional): The colormap to use. Default is 'bwr' (blue-white-red).
    - cell_width (float, optional): Width of each cell in the heatmap. Default is 1.
    - cell_height (float, optional): Height of each cell in the heatmap. Default is 1.
    - vmin (float, optional): Minimum value for colormap scaling.
    - vmax (float, optional): Maximum value for colormap scaling.
    
    Returns:
    - None
    """
    fig_width = len(matrix[0]) * cell_width
    fig_height = len(matrix) * cell_height
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    cax = ax.matshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,alpha=0.8)
    
    # Display the intensity values in each cell
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print_output = round(matrix[i][j],3)
            if ci_matrix is not None:
                ci = ci_matrix[i][j]
                lower = round(print_output-(ci/2),3)
                upper = round(print_output+(ci/2),3)
                print_output = f"{print_output:.3f}\n({lower:.3f}-{upper:.3f})"
                
            ax.text(j, i, str(print_output), ha='center', va='center', color='black',fontsize=12.5)
    
    # Set row and column names
    if row_names:
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names)
    if col_names:
        ax.set_xticks(np.arange(len(col_names)))
        ax.set_xticklabels(col_names, rotation=45, ha='right')
        ax.xaxis.set_ticks_position('bottom')
    
    plt.colorbar(cax)
    plt.show()
    return fig