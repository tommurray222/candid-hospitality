import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def visualise_numeric(df, column, show_boxplot=True, bins=30, color = "color"):
    """
    A function for visualisation of numeric data (e.g. age, expected salary, culture code components) 
    
    Arguments:
        df: of type pd.DataFrame 
        column: of type str, the name of the numeric data column to be plottted
        show_boxplot: -if True, boxplot visual is included (good for cts data like age or salary)
                      -if False, plot only contains histogram (choose if visualising culture code components)
        bins: of type int, number of bins for histogram (for culture code components choose bins =9)
        color: colour of the bars to be plotted
    
    Returns: 
        Either: (default) a histogram/boxplot pair (show_boxplot = True), or a histogram (show_boxplot= False)
    """
    data = df[column].dropna() # removes data that is NaN
    
    # if show_boxplot = True, plot the boxplot and histogram next to each other
    if show_boxplot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # histogram
        sns.histplot(data, kde=True, bins=bins, ax=axes[0], color=color, stat="count")
        axes[0].set_title(f"Distribution of {column}")
        # boxplot
        sns.boxplot(x=data, ax=axes[1], color=color)
        axes[1].set_title(f"Boxplot of {column}")

    # if show_boxplot = False, plot only the histogram
    else:
        # histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data, kde=True, bins=bins, ax=ax, color=color, stat="density")
        ax.set_title(f"Distribution of {column}")

    plt.tight_layout()
    plt.show()
    
#------------------------------------------------------------------------------------------------------------------------------

def visualise_categorical(df, column, top_n=10, show_percent=True, horizontal=True, color = "color"):
    """
    A function for visualising categorical variables (e.g. ethnicity, current city)
    
    Arguments: 
        df: of type pd.DataFrame
        column: of type str, name of the categoric data column to be plotted
        top_n: of type int, the number of categories shown (based on population size 
                                                           e.g. if data = cities, top n=10, -> top 10 cities with top 10 most 
                                                           candidates will be shown, rest will be plotted as "other" )
        show_percent: -if True, percentage is shown alongside bar
                      -if False, only the bar is shown
        horizontal: - (default) if True, the bar chart is displayed horizontally, good for long labels
                    - if False, bar chart is plotted vertically
        color: colour of the bars to be plotted
                    
        Returns:
            Labelled bar chart of data
    """
    series = df[column].dropna()
    counts = series.value_counts()

    # Groups data into top_n + Other
    if len(counts) > top_n:
        top = counts.iloc[:top_n].copy()
        other_sum = counts.iloc[top_n:].sum()
        top["Other"] = other_sum
        counts = top

    # Convert to DataFrame for seaborn
    plot_df = counts.reset_index()
    plot_df.columns = [column, "count"]
    plot_df["percent"] = plot_df["count"] / plot_df["count"].sum() * 100
    
    palette = [color] * len(plot_df)    
    plt.figure(figsize=(10, 6))
    if horizontal:
        ax = sns.barplot(data=plot_df, x="count", y=column, palette=palette)
        if show_percent:
            for p, cnt, pct in zip(ax.patches, plot_df["count"], plot_df["percent"]):
                ax.text(p.get_width() + max(plot_df["count"]) * 0.01,
                        p.get_y() + p.get_height() / 2,
                        f"{cnt} ({pct:.1f}%)", va="center")
        ax.set_xlabel("Count")
        ax.set_ylabel(column)
    else:
        ax = sns.barplot(data=plot_df, x=column, y="count", palette=palette)
        if show_percent:
            for p, cnt, pct in zip(ax.patches, plot_df["count"], plot_df["percent"]):
                ax.text(p.get_x() + p.get_width() / 2,
                        p.get_height() + max(plot_df["count"]) * 0.01,
                        f"{cnt}\n({pct:.1f}%)", ha="center")
        ax.set_ylabel("Count")
        ax.set_xlabel(column)
        plt.xticks(rotation=45)

    ax.set_title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------

def grouped_avgs_plot(df, numeric_col, group_col, stat="mean", title=None, color="color"):
    """
    Works out a summary satistic of numeric data x for categoric data y.
    e.g. x=age , y=department_name, stat="mean" -> average per department

    Arguments:
        df: of type pd.DataFrame
        numeric_col: of form of pd.series or array, should be numeric
        group_col: of form of pd.series or array, should be categoric
        stat: chosen summary statistic (options: mean, median, mode)
        title: of type str, title for grouped averages plot
        
    Returns:
        Plot of "stat" of data x per group y
    """
    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "mode": lambda v: stats.mode(v, keepdims=True).mode[0]
    }
    
    if stat not in stat_funcs:
        raise ValueError(f"Invalid stat '{stat}'. Choose from {list(stat_funcs.keys())}")
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=numeric_col, y=group_col, data=df, estimator=stat_funcs[stat], ci=None, color=color, edgecolor="black")
    
    plt.xlabel(f"{stat.capitalize()} of {numeric_col}")
    plt.ylabel(group_col)
    plt.title(title if title else f"{stat.capitalize()} of {numeric_col} by {group_col}")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

