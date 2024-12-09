from argparse import ArgumentParser
import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from src.actor import DecisionPDF, SwitchPDF

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--human-decision-config-path", type=str, default="configs/human/human.json")
    parser.add_argument("--ai-decision-config-path", type=str, default="configs/ai/ai_ambig.json")
    parser.add_argument("--switch-config-path", type=str, default="configs/switch/switch_ambig.json")
    parser.add_argument("--title", type=str, default="Match on Ambiguous")
    parser.add_argument("--fig-dir", type=str, default="figs")

    args = parser.parse_args()

    with open(args.human_decision_config_path, "r") as f:
        human_decision_config = json.load(f)
    human_pdf = DecisionPDF(**human_decision_config)

    with open(args.ai_decision_config_path, "r") as f:
        ai_decision_config = json.load(f)
    ai_pdf = DecisionPDF(**ai_decision_config)

    with open(args.switch_config_path, "r") as f:
        switch_config = json.load(f)
    switch_pdf = SwitchPDF(**switch_config)

    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(0, 1, 1000)

    pdf_h = human_pdf(x)
    pdf_m = ai_pdf(x)
    pdf_s = switch_pdf(x)
    pdf_hm = pdf_h * (1 - pdf_s) + pdf_m * pdf_s

    ax.plot(x, pdf_h, label=r'$\varphi^{H}$')
    ax.plot(x, pdf_m, label=r'$\varphi^{AI}$')
    ax.plot(x, pdf_hm, label=r'$(1 - \varphi^{S})\varphi^{H} + \varphi^{S}\varphi^{AI}$')
    ax.plot(x, pdf_s, label=r'$\varphi^{S}$', linestyle='--')

    ax.set_xlabel(r'$X$', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title(args.title, fontsize=16)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ai = Path(args.ai_decision_config_path).stem
    switch = Path(args.switch_config_path).stem
    fig_dir = Path(args.fig_dir) 
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_path = fig_dir / f"{ai}_{switch}_decision_pdfs.png"
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)