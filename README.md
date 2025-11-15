# AskJesus Discourse Study

Code and workflow for the project tentatively titled **“The Digitalization of Religion and the Reconstruction of the Sanctity”**, which analyzes how **AskJesus-style Christian discourse** is produced, circulated, and transformed in digital environments.

The project uses **discourse analysis + corpus methods + LLM-assisted tools** to study how “Jesus” is voiced in mediated interactions (videos, captions, Q&A, etc.), how users address this voice, and how notions of **sanctity, authority, and intimacy** are reconstructed through digital mediation.

> **Important**
>
> - This repository is a **research code & workflow repository**.
> - It currently stores **pipeline folders (`01`–`05`) and scripts**, **not** the full raw corpus.
> - `The Digitalization of Religion and the Reconstruction of the Sanctity.pdf` is an **abstract only**, not the full paper.

---

## 1. About this repository

This repo is designed to:

- Organize the **full analysis pipeline** for the AskJesus discourse project:
  - `01_data_collection` → `02_data_preprocessing` → `03_analysis_framework` →  
    `04_implementation` → `05_evaluation`.
- Store **Python code / notebooks** used to:
  - collect and clean text,
  - structure discourse units,
  - apply annotation schemes,
  - generate descriptive statistics and visualizations.
- Provide a **transparent, modular workflow** that can be adapted to other projects on:
  - digital religion,
  - AI-mediated religious talk,
  - or platformized spiritual content.

This is **not** a data-release repository.  
Raw transcripts and platform data are **not** included due to copyright and privacy considerations.

---

## 2. Project overview

### Research questions

The project is guided by questions such as:

- How is the **voice of “Jesus”** constructed in AskJesus-style content?
  - What persona is being performed (gentle counselor, authoritative king, life coach, intimate friend, etc.)?
- How do users **address this voice**?
  - What kinds of needs, fears, and expectations show up in their questions?
- Which **theological motifs** and **emotional scripts** are repeatedly used?
  - suffering, guilt, destiny, spiritual warfare, healing, prosperity, endurance, etc.
- In what ways does digital mediation contribute to the **reconstruction of “the sacred”**?
  - How are sanctity, presence, and authority negotiated when “Jesus” appears as a digital / algorithmic voice?

### Theoretical background (in brief)

The project intersects:

- **Digital religion / mediatization of religion**  
  — how religious authority and presence shift when mediated by platforms and AI.
- **Discourse and narrative analysis**  
  — how stories about God, suffering, and everyday life are patterned and recycled.
- **Framing and cognitive metaphor**  
  — metaphors like *life as battle*, *God as healer/coach/judge*, *sin as debt*, etc.
- **Sanctity and the sacred / profane boundary**  
  — how “the holy” is re-placed into digital formats, and what is lost or reconfigured in the process.

The abstract `The Digitalization of Religion and the Reconstruction of the Sanctity.pdf` summarizes this conceptual angle at a high level.

---

## 3. Repository structure

The project is organized as a **five-stage pipeline** plus the abstract:

```text
AskJesus_Discourse_Study/
├── 01_data_collection/        # scripts & notes for collecting AskJesus-related content
├── 02_data_preprocessing/     # cleaning, segmentation, and structuring of text data
├── 03_analysis_framework/     # coding schemes, label definitions, and analytic design
├── 04_implementation/         # implementation of analysis: scripts, notebooks, models
├── 05_evaluation/             # evaluation, diagnostics, and visualization outputs
├── The Digitalization of Religion and the Reconstruction of the Sanctity.pdf   # abstract only
└── README.md
