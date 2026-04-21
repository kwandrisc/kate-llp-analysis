import matplotlib.pyplot as plt
import numpy as np

loose  = [3499110, 551402, 549865, 113591, 113544, 6]
medium = [2278715, 315934, 314930, 65408, 65389, 78]
tight  = [1070336, 165640, 165191, 27081, 27076, 857]

labels = ["Loose", "Medium", "Tight"]
cuts = [
    "Total",
    "$|eta| \leq 0.8$",
    "chi2/ndf < 3",
    "Outer barrel",
    "High pT",
    "Velo wrms < 1.6"
]

########### helpers ###############

def add_labels(x, y):
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi * 1.2,                 # shift up (important for log scale)
            f"{yi:,}",               # nice formatting with commas
            ha='center',
            va='bottom',
            fontsize=8,
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

    #plt.xticks(x, cuts, rotation=30, ha='right')
    plt.xticks(x, cuts, fontsize=12)
    plt.yscale('log')
    plt.ylabel("Surviving tracks", fontsize=16)
    plt.title("Beam-Induced Background Cutflow")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cutflow.pdf")


def make_mira_cutflow():
    loose_p  = make_pieces(loose)
    medium_p = make_pieces(medium)
    tight_p  = make_pieces(tight)

    all_pieces = np.array([loose_p, medium_p, tight_p])

    # convert to percentages
    totals = np.array([loose[0], medium[0], tight[0]])
    all_pct = all_pieces / totals[:, None] * 100

    stack_labels = [
        "Fail before eta",
        "Fail chi2 after eta",
        "Fail outer barrel after chi2",
        "Fail high pT after outer",
        "Fail velo wrms after high pT",
        "Pass final"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

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
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent of total tracks")
    ax.set_title("100% BIB Stacked Cutflow Comparison")

    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("stacked_cutflow_all.pdf", bbox_inches="tight")
    plt.show()



make_reg_cutflow()
make_mira_cutflow()