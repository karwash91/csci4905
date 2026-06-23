"""Build a simple Project3-style PDF slide deck for the final project."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SLIDE_SIZE = (13.333, 7.5)
IMAGE_DIR = Path("outputs/slide_deck_images")
FONT = "DejaVu Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/final_project_slide_deck.pdf"))
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=Path("outputs/final_project_slide_previews"),
    )
    return parser.parse_args()


def save_broad_shortcut_chart(output_path: Path) -> None:
    labels = ["density only", "CNN normal", "CNN shuffled"]
    accuracy = [0.8764, 0.8940, 0.8896]
    supercritical_recall = [0.984, 0.977, 0.977]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7.4, 4.5))
    plt.bar(x - width / 2, accuracy, width, label="accuracy")
    plt.bar(x + width / 2, supercritical_recall, width, label="supercritical recall")
    plt.ylim(0, 1.0)
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("score")
    plt.title("Broad Dataset Shortcut Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_recall_summary_chart(output_path: Path) -> None:
    labels = [
        "baseline CNN",
        "burned fraction",
        "fixed-density CNN",
        "burn-mask U-Net",
    ]
    supercritical = [0.977, 0.938, 1.000, 0.840]
    critical = [0.809, 0.873, 0.000, 0.747]

    plt.figure(figsize=(7.4, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, supercritical, width, label="supercritical recall")
    plt.bar(x + width / 2, critical, width, label="critical recall")
    plt.xlim(0, 1.0)
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("recall")
    plt.title("Detection Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_chart_assets() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    save_broad_shortcut_chart(IMAGE_DIR / "09_broad_shortcut_chart.png")
    save_recall_summary_chart(IMAGE_DIR / "12_recall_summary_chart.png")


def wrapped_bullets(bullets: list[str], width: int = 44) -> str:
    lines = []
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=width)
        if not wrapped:
            continue
        lines.append(f"• {wrapped[0]}")
        for line in wrapped[1:]:
            lines.append(f"  {line}")
    return "\n".join(lines)


def add_title(fig, title: str, *, size: int = 34, x: float = 0.075, y: float = 0.86) -> None:
    fig.text(x, y, title, fontsize=size, fontname=FONT, weight="bold", va="top")


def add_bullets(
    fig,
    bullets: list[str],
    *,
    x: float,
    y: float,
    size: int = 20,
    width: int = 42,
) -> None:
    fig.text(
        x,
        y,
        wrapped_bullets(bullets, width=width),
        fontsize=size,
        fontname=FONT,
        va="top",
        linespacing=1.22,
    )


def add_image(fig, image_path: Path, box: list[float]) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing slide image: {image_path}")
    ax = fig.add_axes(box)
    ax.axis("off")
    image = plt.imread(image_path)
    ax.imshow(image)


def add_image_caption(fig, caption: str, *, x: float, y: float, width: int = 64) -> None:
    fig.text(
        x,
        y,
        "\n".join(textwrap.wrap(caption, width=width)),
        fontsize=10,
        fontname=FONT,
        va="top",
        color="#333333",
        linespacing=1.15,
    )


def render_title_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    fig.text(
        0.13,
        0.67,
        spec["title"],
        fontsize=42,
        fontname=FONT,
        weight="normal",
        va="top",
    )
    fig.text(
        0.13,
        0.43,
        "\n".join(spec["subtitle"]),
        fontsize=18,
        fontname=FONT,
        va="top",
        linespacing=1.55,
    )
    if spec.get("plain"):
        fig.text(
            0.13,
            0.22,
            wrapped_bullets([spec["plain"]], width=62),
            fontsize=18,
            fontname=FONT,
            va="top",
            linespacing=1.25,
        )
    return fig


def render_question_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])
    add_image(fig, spec["image"], [0.09, 0.16, 0.32, 0.52])
    if spec.get("image_caption"):
        add_image_caption(fig, spec["image_caption"], x=0.09, y=0.145, width=54)
    fig.text(
        0.70,
        0.56,
        spec["question"],
        fontsize=22,
        fontname=FONT,
        ha="center",
        va="center",
        linespacing=1.18,
    )
    if spec.get("plain"):
        fig.text(
            0.49,
            0.24,
            wrapped_bullets([spec["plain"]], width=48),
            fontsize=16,
            fontname=FONT,
            va="top",
            linespacing=1.22,
        )
    if spec.get("caption"):
        fig.text(0.09, 0.08, spec["caption"], fontsize=12, fontname=FONT, va="bottom")
    return fig


def render_bullets_image_slide(spec: dict, *, image_side: str):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])

    if image_side == "right":
        add_bullets(fig, spec["bullets"], x=0.08, y=0.72, width=38)
        add_image(fig, spec["image"], [0.52, 0.15, 0.40, 0.58])
        if spec.get("image_caption"):
            add_image_caption(fig, spec["image_caption"], x=0.52, y=0.115, width=62)
    else:
        add_image(fig, spec["image"], [0.07, 0.16, 0.42, 0.58])
        if spec.get("image_caption"):
            add_image_caption(fig, spec["image_caption"], x=0.07, y=0.125, width=64)
        add_bullets(fig, spec["bullets"], x=0.55, y=0.72, width=36)

    return fig


def render_text_chart_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])
    add_bullets(fig, spec["bullets"], x=0.08, y=0.72, width=38)
    add_image(fig, spec["image"], [0.53, 0.18, 0.38, 0.52])
    if spec.get("image_caption"):
        add_image_caption(fig, spec["image_caption"], x=0.53, y=0.125, width=58)
    return fig


def render_pipeline_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])
    fig.text(
        0.08,
        0.73,
        wrapped_bullets([spec["plain"]], width=78),
        fontsize=18,
        fontname=FONT,
        va="top",
        linespacing=1.2,
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_width = 0.23
    box_height = 0.14
    positions = [
        (0.08, 0.48),
        (0.385, 0.48),
        (0.69, 0.48),
        (0.23, 0.21),
        (0.535, 0.21),
    ]

    for (x, y), step in zip(positions, spec["steps"]):
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.3,
            edgecolor="#222222",
            facecolor="#f7f7f7",
        )
        ax.add_patch(box)
        ax.text(
            x + box_width / 2,
            y + box_height * 0.62,
            step["title"],
            ha="center",
            va="center",
            fontsize=15,
            fontname=FONT,
            weight="bold",
        )
        ax.text(
            x + box_width / 2,
            y + box_height * 0.31,
            "\n".join(textwrap.wrap(step["detail"], width=24)),
            ha="center",
            va="center",
            fontsize=11,
            fontname=FONT,
            linespacing=1.1,
        )

    arrow_pairs = [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3]),
        (positions[3], positions[4]),
    ]
    for start, end in arrow_pairs:
        start_x = start[0] + box_width
        start_y = start[1] + box_height / 2
        end_x = end[0]
        end_y = end[1] + box_height / 2
        if start[1] > end[1]:
            start_x = start[0] + box_width / 2
            start_y = start[1]
            end_x = end[0] + box_width / 2
            end_y = end[1] + box_height
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="#333333",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    return fig


def format_terms(terms: list[tuple[str, str]], *, width: int) -> str:
    lines = []
    for term, definition in terms:
        wrapped = textwrap.wrap(f"{term}: {definition}", width=width)
        if not wrapped:
            continue
        lines.append(wrapped[0])
        for line in wrapped[1:]:
            lines.append(f"  {line}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_glossary_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])

    terms = spec["terms"]
    midpoint = (len(terms) + 1) // 2
    left_terms = terms[:midpoint]
    right_terms = terms[midpoint:]

    fig.text(
        0.08,
        0.74,
        format_terms(left_terms, width=48),
        fontsize=14,
        fontname=FONT,
        va="top",
        linespacing=1.12,
    )
    fig.text(
        0.53,
        0.74,
        format_terms(right_terms, width=48),
        fontsize=14,
        fontname=FONT,
        va="top",
        linespacing=1.12,
    )
    return fig


def render_resources_slide(spec: dict):
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    add_title(fig, spec["title"])
    fig.text(
        0.08,
        0.74,
        spec["left_title"],
        fontsize=20,
        fontname=FONT,
        weight="bold",
        va="top",
    )
    add_bullets(fig, spec["left_bullets"], x=0.08, y=0.67, size=15, width=58)

    fig.text(
        0.56,
        0.74,
        spec["right_title"],
        fontsize=20,
        fontname=FONT,
        weight="bold",
        va="top",
    )
    add_bullets(fig, spec["right_bullets"], x=0.56, y=0.67, size=15, width=48)
    return fig


def render_slide(spec: dict):
    layout = spec["layout"]
    if layout == "title":
        return render_title_slide(spec)
    if layout == "question":
        return render_question_slide(spec)
    if layout == "image_right":
        return render_bullets_image_slide(spec, image_side="right")
    if layout == "image_left":
        return render_bullets_image_slide(spec, image_side="left")
    if layout == "chart_right":
        return render_text_chart_slide(spec)
    if layout == "pipeline":
        return render_pipeline_slide(spec)
    if layout == "resources":
        return render_resources_slide(spec)
    if layout == "glossary":
        return render_glossary_slide(spec)
    raise ValueError(f"Unknown slide layout: {layout}")


def slide_specs() -> list[dict]:
    return [
        {
            "layout": "title",
            "title": "Generative AI for\nSupercritical Forest-Fire Detection",
            "subtitle": [
                "CSCI/DASC 6010",
                "Special Topics - Generative AI",
                "1st Summer 2026",
                "Kareem Washington",
            ],
            "plain": "Plain English: this project asks when a small simulated fire becomes a system-wide burn.",
        },
        {
            "layout": "question",
            "title": "Research Question",
            "question": (
                "Can a model look at the starting forest\n"
                "and predict whether the fire will become\n"
                "supercritical?"
            ),
            "image": IMAGE_DIR / "01_broad_dataset_examples.png",
            "image_caption": "Broad-density simulator examples from scripts/plot_examples.py. Each grid is an initial forest labeled by the final burned fraction after simulation.",
            "plain": "Plain English: supercritical is the dangerous case where the fire reaches almost everything connected to it.",
        },
        {
            "layout": "pipeline",
            "title": "Project Pipeline",
            "plain": "Plain English: I generate a forest, simulate the fire, train models, then test whether the models learned the right thing.",
            "steps": [
                {
                    "title": "1. Simulate",
                    "detail": "NumPy forest-fire grids with measured burn outcomes",
                },
                {
                    "title": "2. Generate",
                    "detail": "Conditional VAE creates requested forest layouts",
                },
                {
                    "title": "3. Check",
                    "detail": "Generated grids are burned again in the simulator",
                },
                {
                    "title": "4. Detect",
                    "detail": "CNN predicts subcritical, critical, or supercritical",
                },
                {
                    "title": "5. Stress Test",
                    "detail": "Density, shuffling, and burn-mask tests check the story",
                },
            ],
        },
        {
            "layout": "image_right",
            "title": "Background: Criticality",
            "bullets": [
                "Plain English: some systems sit near a tipping point where a small event can cause a large chain reaction",
                "Project 3 framed this as self-organizing behavior: a forest can build up until one spark spreads widely",
                "Here, supercritical means burned fraction is at least 0.90",
                "Main goal: detect that big-burn outcome from the starting grid",
            ],
            "image": IMAGE_DIR / "02_fixed_density_examples.png",
            "image_caption": "Fixed-density examples from scripts/generate_spatial_dataset.py. Every grid has density 0.55, so visible layout differences drive the burn outcome.",
        },
        {
            "layout": "image_right",
            "title": "Simulation Setup",
            "bullets": [
                "Plain English: I use a simple grid world so the answer can be measured exactly",
                "Cells are empty, tree, burning, or burned",
                "Fire spreads up, down, left, and right until no trees are burning",
                "Labels come from final burned fraction",
            ],
            "image": IMAGE_DIR / "01_broad_dataset_examples.png",
            "image_caption": "Generated by the NumPy forest-fire simulator. Rows show starting grids whose simulated outcomes become subcritical, critical, or supercritical.",
        },
        {
            "layout": "image_right",
            "title": "Generative Add-On",
            "bullets": [
                "Plain English: the generator makes synthetic grids, then the simulator checks if they behave as requested",
                "Conditional VAE requested subcritical, critical, or supercritical grids",
                "Project 2 lesson: generated outputs need visual and quantitative checks",
                "119 of 120 generated samples matched their requested class",
                "Caveat: broad dataset still has a density shortcut",
            ],
            "image": IMAGE_DIR / "00b_conditional_vae_match_matrix.png",
            "image_caption": "Generated-sample check from scripts/train_conditional_vae.py. The matrix compares requested class against the class produced by running the simulator.",
        },
        {
            "layout": "image_right",
            "title": "Baseline Supercritical Detector",
            "bullets": [
                "Plain English: the first CNN was very good at spotting big-burn cases",
                "Test accuracy: 0.8940",
                "Supercritical recall: 0.977",
                "Project 1 lesson: high accuracy still needs a shortcut check",
            ],
            "image": IMAGE_DIR / "03_baseline_cnn_confusion_matrix.png",
            "image_caption": "Confusion matrix from scripts/train_cnn.py on the broad-density test split. The bottom-right cell shows supercritical examples correctly detected.",
        },
        {
            "layout": "chart_right",
            "title": "Density Shortcut Test",
            "bullets": [
                "Plain English: if shuffling trees does not hurt, the model is not using layout much",
                "Density-only prediction was almost as strong as the CNN",
                "Shuffled CNN still detected supercritical cases well",
                "The broad dataset was mostly density-driven",
            ],
            "image": IMAGE_DIR / "09_broad_shortcut_chart.png",
            "image_caption": "Chart generated inside scripts/build_slide_deck.py from ablation results. It compares density-only, normal CNN, and shuffled-grid CNN performance.",
        },
        {
            "layout": "image_left",
            "title": "Fixed-Density Challenge",
            "bullets": [
                "Plain English: all forests have the same number of trees, so placement has to matter",
                "Density fixed at 0.55",
                "Density-only prediction dropped to chance",
                "CNN stayed above chance, but shuffled CNN dropped back to chance",
            ],
            "image": IMAGE_DIR / "06_fixed_density_cnn_confusion_matrix.png",
            "image_caption": "Confusion matrix from training the baseline CNN on data/spatial_64. Density is fixed, so the model must use tree placement rather than tree count.",
        },
        {
            "layout": "pipeline",
            "title": "Burn-Mask U-Net",
            "plain": "Plain English: instead of guessing the final class first, this model predicts the burned region first.",
            "steps": [
                {
                    "title": "1. Input",
                    "detail": "two grids: tree locations and starting fire cells",
                },
                {
                    "title": "2. U-Net",
                    "detail": "image-to-image CNN keeps local layout information",
                },
                {
                    "title": "3. Burn Mask",
                    "detail": "model predicts which cells are likely to burn",
                },
                {
                    "title": "4. Fraction",
                    "detail": "predicted burned cells become a burned fraction",
                },
                {
                    "title": "5. Class",
                    "detail": "fraction becomes subcritical, critical, or supercritical",
                },
            ],
        },
        {
            "layout": "image_right",
            "title": "Burn-Mask Results",
            "bullets": [
                "Plain English: instead of guessing a label, the model predicts which cells burn",
                "This matches how a person would solve it: trace connected trees",
                "Supercritical recall stayed strong: 63 of 75",
                "It also recovered many middle critical cases the direct CNN missed",
            ],
            "image": IMAGE_DIR / "07_burn_mask_unet_confusion_matrix.png",
            "image_caption": "Derived class matrix from scripts/train_burn_mask_unet.py. The U-Net predicts a burn mask first, then burned fraction is converted to a class.",
        },
        {
            "layout": "chart_right",
            "title": "Final Takeaways",
            "bullets": [
                "Plain English: supercritical detection is achievable, but the test setup matters",
                "Previous projects shaped the method: compare models, check generated samples, control density",
                "Density can make results look better than they are",
                "Best direction for spatial reasoning: predict burn area first, then classify risk",
            ],
            "image": IMAGE_DIR / "12_recall_summary_chart.png",
            "image_caption": "Summary chart generated inside scripts/build_slide_deck.py. It compares supercritical recall with critical recall across the main model variants.",
        },
        {
            "layout": "resources",
            "title": "Resources",
            "left_title": "Project Resources",
            "left_bullets": [
                "Code and results are in csci4905-final",
                "Simulator: src/forest_fire_sim.py",
                "Generator: scripts/train_conditional_vae.py",
                "Classifiers: scripts/train_cnn.py and scripts/train_burn_mask_unet.py",
                "Report and notes: docs/final_report_draft.md and docs/research_notes.md",
                "Codex helped brainstorm, debug, and organize experiments",
            ],
            "right_title": "Research Background",
            "right_bullets": [
                "Clar, Drossel, Schwabl: forest-fire self-organized criticality",
                "Schenk, Drossel, Clar, Schwabl: finite-size effects in forest-fire models",
                "Newman and Ziff: square-lattice percolation threshold near 0.5927",
                "Carrasquilla and Melko: neural networks can learn phase transitions",
                "Zhang, Liu, Wei: machine learning for percolation and related transitions",
            ],
        },
        {
            "layout": "glossary",
            "title": "Glossary",
            "terms": [
                ("Generative AI", "a model that creates new examples instead of only labeling existing ones"),
                ("CNN", "convolutional neural network; a model commonly used for images or grid data"),
                ("VAE", "variational autoencoder; a generative model that learns a compressed latent space"),
                ("U-Net", "an image-to-image CNN often used to predict masks or segment regions"),
                ("Mask", "a grid marking which cells belong to something, such as burned cells"),
                ("Recall", "out of all true examples of a class, the fraction the model correctly finds"),
                ("Confusion matrix", "a table comparing actual classes against predicted classes"),
                ("Density / burned fraction", "density is how many cells start as trees; burned fraction is how much burns by the end"),
                ("Supercritical", "the big-burn case; here, burned fraction is at least 0.90"),
                ("Ablation", "a test where one factor is removed or changed to see what the model uses"),
                ("Shortcut", "an easy pattern the model uses that may not be the intended reasoning"),
                ("Self-organized criticality", "a system naturally builds toward a tipping point where one small event can spread widely"),
            ],
        },
    ]


def build_deck(output_path: Path, png_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    build_chart_assets()
    for old_preview in png_dir.glob("slide_*.png"):
        old_preview.unlink()

    specs = slide_specs()
    with PdfPages(output_path) as pdf:
        for slide_number, spec in enumerate(specs, start=1):
            fig = render_slide(spec)
            pdf.savefig(fig)
            fig.savefig(png_dir / f"slide_{slide_number:02d}.png", dpi=150)
            plt.close(fig)


def write_markdown_outline(path: Path) -> None:
    lines = ["# Final Project Slide Deck", "", "PDF deck: `outputs/final_project_slide_deck.pdf`", ""]
    for number, spec in enumerate(slide_specs(), start=1):
        lines.append(f"## {number}. {spec['title'].replace(chr(10), ' ')}")
        lines.append("")
        if spec.get("image"):
            lines.append(f"Image: `{spec['image']}`")
            if spec.get("image_caption"):
                lines.append(f"Caption: {spec['image_caption']}")
            lines.append("")
        if spec.get("question"):
            lines.append(spec["question"].replace("\n", " "))
            lines.append("")
        if spec.get("plain"):
            lines.append(f"- {spec['plain']}")
        for step in spec.get("steps", []):
            lines.append(f"- **{step['title']}:** {step['detail']}")
        if spec.get("left_title"):
            lines.append(f"### {spec['left_title']}")
            for bullet in spec.get("left_bullets", []):
                lines.append(f"- {bullet}")
        if spec.get("right_title"):
            lines.append(f"### {spec['right_title']}")
            for bullet in spec.get("right_bullets", []):
                lines.append(f"- {bullet}")
        for bullet in spec.get("bullets", []):
            lines.append(f"- {bullet}")
        for term, definition in spec.get("terms", []):
            lines.append(f"- **{term}:** {definition}")
        if spec.get("subtitle"):
            lines.extend(f"- {line}" for line in spec["subtitle"])
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    build_deck(args.output, args.png_dir)
    write_markdown_outline(Path("docs/final_project_slide_deck.md"))
    print(f"Saved {args.output}")
    print(f"Saved previews to {args.png_dir}")
    print("Saved docs/final_project_slide_deck.md")


if __name__ == "__main__":
    main()
