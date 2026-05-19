import matplotlib.pyplot as plt
import numpy as np

loose  = [3499110, 551402, 549865, 113591, 113544, 6]
medium = [2278715, 315934, 314930, 65408, 65389, 78]
tight  = [1070336, 165640, 165191, 27081, 27076, 857]

loose_arr = np.array(loose, dtype=float)
medium_arr = np.array(medium, dtype=float)
tight_arr = np.array(tight, dtype=float)

windows = ["Loose", "Medium", "Tight"]
cuts = [
    "Total",
    r"$|\eta| \leq 0.8$",
    r"$\chi^2/ndf < 3$",
    "Hit req",
    r"$p_T < 10 TeV$",
    "W.RMS < 1.6"
]

cuts_efficiency = [
        "eta / total",
        "chi2 / eta",
        "outer / chi2",
        "high pT / outer",
        "velo wrms / high pT",
        "final / total"
    ]

########### helpers ###############

def add_labels(x, y):
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi * 1.2, # shifting up - see if still helpful with larger fontsize              
            f"{yi:,}", # adding commas               
            ha='center',
            va='bottom',
            fontsize=12,
            rotation=0
        )

def make_pieces(arr):
    total, eta, chi2, outer, highpt, wrms = arr
    return [
        total - eta,
        eta - chi2,
        chi2 - outer,
        outer - highpt,
        highpt - wrms,
        wrms
    ]


def per_cut_eff(counts):
    total, eta, chi2, outer, highpt, wrms = counts
    return np.array([
        eta / total,
        chi2 / eta,
        outer / chi2,
        highpt / outer,
        wrms / highpt,
        wrms / total
    ])


########### plotting code ###############


def make_reg_cutflow():
    x = np.arange(len(cuts))

    plt.figure(figsize=(10, 6))

    plt.plot(x, loose,  marker='o', linewidth=2, label='Loose window')
    plt.plot(x, medium, marker='o', linewidth=2, label='Medium window')
    plt.plot(x, tight,  marker='o', linewidth=2, label='Tight window')

    # add_labels(x, loose)
    # add_labels(x, medium)
    # add_labels(x, tight)

    plt.xticks(x, cuts, rotation=20, ha='right', fontsize=20)
    plt.yticks(fontsize=16)
    plt.tick_params(axis="both", length=8, width=1.5, labelsize=15)
    plt.yscale('log')
    plt.ylabel("Surviving tracks (log scale axis)", fontsize=20)
    plt.title("100% BIB Track-Level Cutflow", fontsize=20)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig("pdf/cutflow.pdf")
    print("cutflow diagram saved to pdf/cutflow.pdf")


def make_mira_cutflow():
    loose_p  = make_pieces(loose)
    medium_p = make_pieces(medium)
    tight_p  = make_pieces(tight)

    all_pieces = np.array([loose_p, medium_p, tight_p])

    # convert to percentages
    totals = np.array([loose[0], medium[0], tight[0]])
    all_pct = all_pieces / totals[:, None] * 100

    stack_labels = [
        "Fail eta",
        "Fail chi2 after eta",
        "Fail outer barrel after chi2",
        "Fail high pT after outer",
        "Fail velo wrms after high pT",
        "Pass final"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(windows))
    bottom = np.zeros(len(windows))

    for i in range(len(stack_labels)):
        values = all_pct[:, i]
        ax.bar(x, values, bottom=bottom, label=stack_labels[i])
        
        # OPTIONAL: add labels inside bars (only if big enough)
        for j in range(len(x)):
            if values[j] > 2:  # avoid clutter
                ax.text(
                    x[j],
                    bottom[j] + values[j]/2,
                    f"{values[j]:.1f}%",
                    ha='center',
                    va='center',
                    fontsize=8
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Percent of total tracks")
    ax.set_title("100% BIB Stacked Cutflow Comparison")

    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("pdf/stacked_cutflow_all.pdf", bbox_inches="tight")
    plt.show()


def efficiency_heatmap():
    
    data_eff = np.vstack([
        per_cut_eff(loose_arr),
        per_cut_eff(medium_arr),
        per_cut_eff(tight_arr)
    ]) * 100  # convert to percent

    data_rej = 100 - data_eff

    fig, ax = plt.subplots(figsize=(10, 3.8))

    im = ax.imshow(data_rej, aspect="auto", cmap="viridis")

    # axis labels
    ax.set_xticks(np.arange(len(cuts_efficiency)))
    ax.set_xticklabels(cuts_efficiency, rotation=10, ha="right")
    ax.set_yticks(np.arange(len(windows)))
    ax.set_yticklabels(windows)

    ax.set_title("100% BIB Cutflow Step Rejection Heatmap")
    ax.set_ylabel("Timing window")

    # write values in each cell
    for i in range(data_rej.shape[0]):
        for j in range(data_rej.shape[1]):
            val = data_rej[i, j]
            text_color = "white" if val < 50 else "black"
            if val >= 0.1:
                label = f"{val:.2f}%"
            else:
                label = f"{val:.3f}%"
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Step rejection (%)")

    plt.tight_layout()
    plt.savefig("pdf/bib_cutflow_heatmap.pdf", bbox_inches="tight")
    plt.show()


def cut_efficiencies_bar():
    loose_eff  = per_cut_eff(loose) * 100
    medium_eff = per_cut_eff(medium) * 100
    tight_eff  = per_cut_eff(tight) * 100

    loose_rej  = 100 - loose_eff
    medium_rej = 100 - medium_eff
    tight_rej  = 100 - tight_eff
    
    x = np.arange(len(cuts_efficiency))
    w = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w, loose_rej,  width=w, label="Loose")
    bars2 = ax.bar(x,     medium_rej, width=w, label="Medium")
    bars3 = ax.bar(x + w, tight_rej,  width=w, label="Tight")

    ax.set_xticks(x)
    ax.set_xticklabels(cuts_efficiency)
    ax.set_ylabel("Rejection (%)")
    ax.set_title("100% BIB Cutflow Step Rejection")
    ax.legend()

    # add labels
    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2,
                h + 1,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )


    plt.tight_layout()
    plt.savefig("pdf/bib_per_cut_rejections.pdf", bbox_inches="tight")
    plt.show()


make_reg_cutflow()
make_mira_cutflow()
efficiency_heatmap()
cut_efficiencies_bar()