# Interview Prep — Lung Cancer CT Classification (Data Analyst Role)

This file is built around your **three resume bullets**. A data analyst interviewer will care less about how ResNet works inside and more about **how you read data, chose metrics, found a problem and showed that the fix worked**. For the deep-learning details, see `study.md`.

All numbers below come from the saved outputs in `Lung Cancer Pred.ipynb`.

---

## 0. Read This First — 4 Things That Could Trip You Up

1. **Your resume says "~83%". The exact figure is 83.17%** (262 of 315 test images correct). The baseline was 81.27% (256/315). So the gain is **+1.9 points, which is 6 more images correct.** Say that plainly. An analyst who knows the absolute count of what changed earns trust.
2. **That gain is inside the noise.** With n = 315, the 95% confidence interval on accuracy is about **±4.1 points** (√(0.83·0.17/315) ≈ 0.021 → ×1.96). If someone asks "is that significant?", the honest answer is *"Not on its own. The stronger evidence is that Kappa, MCC and AUC all went up together, and large-cell recall rose by 13 points."* (See Q12.)
3. **More than one thing changed between runs.** The improved run added class weights **and** the scheduler **and** removed hue/saturation jitter, which is meaningless on grayscale CT. You can't credit the gain to one change alone. If asked, say so and propose an ablation: change one thing at a time.
4. **Don't mention the ensemble, EfficientNet, ConvNeXt or AMP** unless you have run them and have the results. `study.md` describes them, but the notebook only contains the ResNet50 run, and **they are not on your resume**. Anything you bring up can be probed.

---

## 1. The 60-Second Answer to "Walk me through this project"

> "I built a classifier for chest CT scans with four classes: three lung cancer subtypes plus normal. The dataset had only about 1,000 images, so instead of training from scratch I fine-tuned an ImageNet-pretrained ResNet50. I froze most of the network and retrained only the last block and a new classification head.
>
> The baseline reached 81.3% accuracy. When I looked at the confusion matrix, the errors weren't random. The model never mixed up cancer and normal, but it struggled with the subtypes, and large-cell carcinoma, the smallest class, had only 75% recall. The class counts confirmed an imbalance: 195 adenocarcinoma training images against 115 large-cell. So I weighted the loss inversely to class frequency and added a learning-rate scheduler. Accuracy went to 83.2%, and large-cell recall went from 75% to 88%.
>
> Because the classes are imbalanced, I didn't rely on accuracy alone. I reported Cohen's Kappa, MCC and per-class specificity. Kappa went from 0.74 to 0.77 and MCC from 0.75 to 0.78."

The structure is **problem → baseline → diagnosis from the data → targeted fix → measured result → right metrics**. That is how an analyst thinks, and that is what this story shows.

---

## 2. Numbers to Know Cold

### Data (the analyst's first question is always "what does the data look like?")

| Split | Adeno | Large cell | Normal | Squamous | Total |
|---|---|---|---|---|---|
| Train | 195 | **115** (smallest) | 148 | 155 | 613 |
| Valid | 23 | 21 | 13 | 15 | 72 |
| Test | **120** | 51 | 54 | 90 | 315 |

- Imbalance ratio in train: 195 / 115 ≈ **1.7 : 1**. That is moderate, not extreme.
- The test set is **more** skewed (adeno is 38% of test vs. 32% of train). Worth mentioning: the test distribution differs from the train distribution.
- The split is roughly **61 / 7 / 32**, which is unusual. The validation set is tiny (72 images), so validation loss is noisy, and that affects checkpointing and the scheduler.
- Class weights used: adeno **0.786**, large cell **1.333**, normal **1.035**, squamous **0.989**. The formula is `N / (K × count_k)`, the same as sklearn's `"balanced"` setting.

### Baseline vs. improved

| Metric | Baseline | Improved | Change |
|---|---|---|---|
| Accuracy | 81.27% | **83.17%** | +1.9 pts (6 images) |
| Weighted AUC (one-vs-rest) | 0.946 | **0.959** | +0.013 |
| Cohen's Kappa | 0.74 | **0.769** | +0.03 |
| MCC | 0.75 | **0.777** | +0.03 |
| Macro F1 | — | **0.85** | |

### Per-class results for the improved model (test set)

| Class | Precision | Recall | F1 | Specificity | Support |
|---|---|---|---|---|---|
| Adenocarcinoma | 0.89 | **0.68** | 0.77 | 0.949 | 120 |
| Large cell | 0.80 | **0.88** (was 0.75) | 0.84 | 0.958 | 51 |
| Normal | **1.00** | **1.00** | 1.00 | 1.000 | 54 |
| Squamous | 0.72 | 0.91 | 0.80 | **0.858** | 90 |

**The insights to be able to state:**
- **Cancer vs. normal is perfect.** Every error is a confusion *between cancer subtypes*. Clinically, that is much less dangerous than missing a cancer.
- **The main remaining error is adenocarcinoma predicted as squamous.** About 38 of 120 adeno scans were missed, mostly into squamous. That explains squamous's low precision (0.72) and low specificity (0.858): it acts as a "sink" for adeno errors.
- **The class weighting did what it should, and had a cost.** Large cell (upweighted ×1.33) gained 13 points of recall. Adeno (downweighted ×0.79) *lost* 2 points of recall (0.70 → 0.68). Reweighting moves errors between classes; it doesn't remove them for free. **Saying this unprompted is a strong signal.**
- **The adeno/squamous confusion is not caused by imbalance.** Adeno is the *largest* class. That confusion is more likely because the two subtypes genuinely look alike on CT. So be precise: imbalance explained the **large-cell** problem; similar visual features explain the **adeno/squamous** problem.

---

## 3. Metrics — Definitions *and* Formulas (the most-tested area for an analyst)

Binary confusion matrix terms: TP, FP, FN, TN. For multi-class, compute them per class with a **one-vs-rest** approach.

| Metric | Formula | Plain English | Why it matters here |
|---|---|---|---|
| Accuracy | (TP+TN)/all | % correct | Misleading when classes are imbalanced |
| Precision | TP/(TP+FP) | When I say X, how often am I right? | Cost of false alarms |
| Recall / Sensitivity | TP/(TP+FN) | Of all real X, how many did I catch? | **Missed cancer = worst error** |
| Specificity | TN/(TN+FP) | Of all non-X, how many did I correctly leave alone? | Unneeded biopsies; sklearn has no built-in for it, so **you computed it from the confusion matrix** |
| F1 | 2PR/(P+R) | Balance of precision and recall | Harmonic mean, so a low value in either drags it down |
| Cohen's Kappa | (pₒ − pₑ)/(1 − pₑ) | Agreement beyond what chance would give | Corrects for chance agreement driven by class frequency |
| MCC | (TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Correlation between predictions and truth, from −1 to +1 | Uses all four cells, so it stays reliable under imbalance |
| ROC-AUC | Area under the TPR-vs-FPR curve | Probability that a random positive is ranked above a random negative | Doesn't depend on the threshold |

**Know how to explain these:**
- **Kappa scale (Landis & Koch):** <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, **0.61–0.80 substantial** (yours: 0.77), >0.80 almost perfect.
- **pₑ in Kappa** is the agreement you'd expect if predictions were random but kept the same class proportions. For each class, multiply (share of actual) × (share of predicted), then sum.
- **Macro vs. weighted average:** macro gives every class equal weight, so it treats rare classes fairly. Weighted averages by support, so it's dominated by the big classes. Your macro F1 (0.85) is *higher* than weighted F1 (0.83) because the smaller classes (normal, large cell) perform better than the biggest one (adeno).
- **Why not report only accuracy?** A model that predicts "adenocarcinoma" for every test image scores 38% (120/315) with zero skill. That model would get Kappa = 0 and MCC = 0. Those metrics expose it; accuracy doesn't.
- **Kappa vs. MCC:** both correct for chance. MCC is generally considered more reliable on imbalanced data. Kappa can behave oddly when the marginal distributions are very uneven (the "Kappa paradox"). Reporting both, and seeing that they agree, is extra reassurance.
- **One-vs-rest (OvR) AUC:** for each class, treat it as positive and all others as negative, compute AUC, then average weighted by support.

---

## 4. Most Probable Interview Questions (with answers)

### A. The project and the data

**Q1. Tell me about this project.** → Use section 1.

**Q2. Where did the data come from, and what did you check before modelling?**
It's the public Kaggle "Chest CT-Scan images" dataset, about 1,000 images already split into train, validation and test. Before modelling I checked the **class counts per split** (that's where I saw the 195 vs. 115 imbalance), confirmed the labels were consistent across splits, and noticed the label names include TNM staging codes (`T2_N0_M0_Ib`). Those codes are part of the folder name, not something I predict. I also noticed the scans are grayscale, which is why I removed color augmentations.

**Q3. What data-quality risks exist in this dataset?**
- **Possible patient leakage:** if slices from the same patient appear in both train and test, test accuracy is inflated. The dataset doesn't provide patient IDs, so I can't rule it out. This is the biggest caveat.
- **Single source:** one dataset, unknown scanners and hospitals, so generalization is unproven.
- **Label noise:** I can't verify the clinical ground truth.
- **Every cancer class has one staging code**, so the model may partly learn stage- or location-specific features rather than the subtype.
- **Small validation set** (72 images), so decisions based on validation loss are noisy.

**Q4. Why is your split 61/7/32 instead of something like 70/15/15?**
It came pre-split from Kaggle, and I kept it so my results are comparable to other work on the same data. If I rebuilt it, I'd use a **stratified** split (same class proportions in each split) and a larger validation set, ideally grouped by patient.

### B. The imbalance diagnosis (bullet 2, the most likely deep-dive)

**Q5. How did you know class imbalance was the problem?**
Two pieces of evidence that pointed the same way. First, the **class counts**: 195 adeno vs. 115 large cell in training. Second, the **confusion matrix and per-class recall**: large cell, the smallest class, had the lowest recall at 0.75. An unweighted loss rewards the model for getting common classes right, so rare classes get under-predicted. After weighting, large-cell recall went to 0.88, which confirms the diagnosis.

*Be ready for the follow-up:* "But your biggest error was adeno → squamous, and adeno is the *majority* class." → "Right. That confusion isn't explained by imbalance. It's more likely the two subtypes look similar on CT, which even radiologists often confirm only by biopsy. Weighting fixed the imbalance problem; the adeno/squamous boundary needs a different fix, such as more data, higher resolution, or interpretability work to see what the model is looking at."

**Q6. What does class-weighted cross-entropy actually do?**
It multiplies each sample's loss by a weight for its class. I used `N / (K × count)`, so rare classes get weight above 1 (large cell ≈ 1.33) and common ones below 1 (adeno ≈ 0.79). A mistake on a rare class then costs more, and the model can't reduce its loss just by favoring the majority.

**Q7. What other ways could you handle imbalance?**
- **Oversampling** the minority class (for example PyTorch `WeightedRandomSampler`) or **undersampling** the majority.
- **Data augmentation** targeted at the minority classes.
- **Focal loss**, which focuses on hard examples.
- **Threshold tuning:** change the decision cutoff per class after training.
- **Collect more data** for the rare classes, which is usually the best fix.
- *SMOTE* works for tabular data but doesn't make much sense for raw images.
Why weighting: it's one line, doesn't change the data pipeline, and doesn't duplicate images (duplicating images raises overfitting risk on a small dataset).

**Q8. Did the fix have any downside?**
Yes. Adeno recall dropped slightly (0.70 → 0.68) because I downweighted it. Reweighting shifts the trade-off between classes. The net effect was positive (accuracy, Kappa, MCC and AUC all rose), but it wasn't free.

**Q9. What is ReduceLROnPlateau and why did you add it?**
It cuts the learning rate (I used ×0.5) when validation loss hasn't improved for a set number of epochs (patience = 3). Validation loss was oscillating in later epochs, which suggests the steps were too big to settle into a good minimum. Smaller steps late in training help it converge.

**Q10. How did you pick the final model?**
I saved a checkpoint whenever validation loss hit a new low, and evaluated that checkpoint rather than the last epoch. Training loss kept falling while validation loss flattened, which is overfitting. The checkpoint captures the model at its best generalization point. The test set was used **once**, at the end.

### C. Metrics (bullet 3, analyst core)

**Q11. Why Kappa and MCC instead of accuracy?** → Section 3: the "predict everything as adeno" example gives 38% accuracy with Kappa and MCC of 0.

**Q12. Is 81.3% → 83.2% a real improvement or noise?**
On accuracy alone I can't claim it's significant: it's 6 images out of 315, and the 95% CI is about ±4 points. What makes me more confident is that **every metric moved in the same direction** (AUC, Kappa, MCC), and the targeted metric, large-cell recall, moved a lot (+13 points). To test it properly I'd use **McNemar's test**, which compares two models on the *same* test images by looking only at the cases where they disagree. I'd also run **k-fold cross-validation** or bootstrap the test set to get confidence intervals, and repeat training with several random seeds.

**Q13. Why compute specificity separately?**
sklearn's `classification_report` doesn't include it. I computed it per class from the confusion matrix: TN is the sum of everything outside that class's row and column, and FP is the column total minus the diagonal cell. In medicine it matters because low specificity means false alarms, which lead to unnecessary biopsies and patient anxiety.

**Q14. Which error is worse here, a false positive or a false negative?**
A false negative: telling someone with cancer that they're healthy delays treatment. A false positive costs a follow-up scan. My model had **zero** cancer→normal errors. In deployment I would set thresholds to favor **sensitivity** over specificity.

**Q15. Which single metric would you report to a hospital manager?**
I wouldn't give one number. I'd lead with **sensitivity for cancer vs. normal** (the "did we miss anyone" question, 100% here), then per-subtype recall, then say clearly that the test set was small. For a technical audience: MCC plus per-class recall.

**Q16. What does an AUC of 0.959 mean in plain words?**
If I pick one scan that belongs to a class and one that doesn't, the model gives a higher score to the correct one about 96% of the time. It measures ranking quality across all thresholds, not accuracy at one threshold.

**Q17. Explain the confusion matrix to a non-technical stakeholder.**
"Rows are what the scan really was, columns are what the model said. Numbers on the diagonal are correct calls. Everything off the diagonal is a mistake, and *where* it sits tells you which kind. Here, the normal row and column are perfectly clean, so the model never told a cancer patient they were fine. Most of the mistakes sit in one cell: adenocarcinoma called squamous."

### D. The modelling (bullet 1, expect lighter questions for an analyst role)

**Q18. What is transfer learning and why use it?**
Start with a model already trained on a huge dataset (ImageNet, about 1.2M images) and adapt it. Its early layers already detect edges and textures, which are useful for any image. With about 1,000 images, training 25M parameters from scratch would overfit badly.

**Q19. What does "selective layer freezing" mean?**
I turned off gradient updates for all layers except the last residual block (`layer4`) and my new head. Early layers stay general; only the deepest, most task-specific layers adapt. Fewer trainable parameters means less overfitting on small data.

**Q20. What is the custom head?**
`Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→4)`. It replaces ImageNet's 1000-class output with my 4 classes. Dropout randomly switches off half the neurons during training to fight overfitting.

**Q21. Why two learning rates?**
The pretrained layer gets a small one (1e-5) so its good weights aren't destroyed. The new, randomly initialized head gets a 10× larger one (1e-4) so it learns quickly.

**Q22. How did you prevent overfitting?**
Freezing most layers, dropout, data augmentation (rotation, flip, small shifts and zoom, mild brightness and contrast), checkpointing on validation loss, and the LR scheduler.

**Q23. Why remove the color augmentation?**
CT scans are grayscale. Hue and saturation jitter do nothing useful, so I kept only brightness and contrast. This is a data-understanding decision, not a modelling one.

### E. Analyst-framed and behavioral questions

**Q24. You're a data analyst. Why does a deep learning project belong on your resume?**
"The model itself is a tool. The skills I'd bring to this role are the analytical ones: exploring the data before modelling, finding the failure pattern in a confusion matrix, choosing metrics that fit an imbalanced problem, and being honest about what the numbers can and can't support. That's the same workflow I'd use on a churn model or a sales KPI."

**Q25. How would you present these results in a dashboard?**
- A KPI row with sensitivity (cancer vs. normal), macro F1, MCC and test n.
- A **confusion-matrix heatmap** as the main visual.
- A per-class recall/precision bar chart comparing baseline and improved.
- ROC curves per class.
- A clearly visible note about the small sample size and confidence intervals.
Tools: Tableau, Power BI, or Python (matplotlib/seaborn, which I used for the heatmap and ROC curves).

**Q26. What would you do differently / next?**
1. **Group the split by patient** to rule out leakage.
2. **Ablation study:** change one factor at a time to isolate what caused the gain.
3. **Cross-validation and bootstrap CIs**, plus McNemar's test.
4. **Grad-CAM heatmaps** to see *where* the model looks, which builds trust and helps explain the adeno/squamous confusion.
5. **External validation** on data from another hospital.
6. **Threshold tuning** to favor sensitivity.

**Q27. What was the hardest part?**
"Resisting the urge to just tune until the number went up. The baseline's 81% looked decent, but breaking it down by class showed the weak point was large-cell recall, not overall accuracy. Once I focused on that one question, the fix was obvious and I could check whether it worked."

**Q28. Would this be deployed in a hospital?**
No. It's a small, single-source dataset with no external validation, no check for patient-level leakage and no regulatory clearance. The realistic use is a research prototype, or at most a triage aid that flags scans for a radiologist.

---

## 5. Likely Follow-Up "Gotchas" — One-Line Answers

| If they ask… | Say… |
|---|---|
| "So the model is 83% accurate?" | "On this 315-image test set, yes, with about ±4 points of uncertainty." |
| "Why is squamous precision so low?" | "It absorbs most of the misclassified adenocarcinoma scans: about 32 false positives." |
| "Why is macro F1 higher than weighted F1?" | "The biggest class (adeno) has the weakest F1, so weighting by support pulls the average down." |
| "What is the null/baseline accuracy?" | "38%, from always predicting the majority test class (adeno)." |
| "Did you tune on the test set?" | "No. Checkpointing and the scheduler used validation loss only; the test set was touched once." |
| "Is 1,000 images enough?" | "For training from scratch, no. For fine-tuning a pretrained model, it's workable, which is the point of transfer learning." |
| "Precision or recall for cancer?" | "Recall. A missed cancer costs far more than a false alarm." |
| "Kappa of 0.77, is that good?" | "'Substantial' agreement on the Landis–Koch scale (0.61–0.80)." |

---

## 6. General Concepts to Brush Up On (in case the questions widen)

- **Stats:** confidence intervals for a proportion, hypothesis testing and p-values, McNemar's test, bootstrap, the bias–variance trade-off, Type I vs. Type II error (FP vs. FN).
- **Validation:** train/val/test roles, k-fold vs. stratified k-fold, data leakage (and **group** leakage, e.g. by patient), overfitting vs. underfitting.
- **Metrics:** the full confusion-matrix family, ROC vs. **Precision–Recall curves** (PR is more informative under heavy imbalance), macro vs. micro vs. weighted averaging, calibration.
- **Imbalance toolkit:** class weights, over/undersampling, SMOTE (tabular data only), threshold moving, choosing the right metric.
- **Python stack:** pandas, NumPy, scikit-learn (`classification_report`, `confusion_matrix`, `cohen_kappa_score`, `matthews_corrcoef`, `roc_auc_score`, `label_binarize`), matplotlib/seaborn.
- **Communication:** explain every metric in one sentence to a non-technical person. Practice Q16 and Q17 out loud.

---

## 7. Final Checklist Before the Interview

- [ ] Can say the 60-second pitch without notes.
- [ ] Know the exact numbers: **81.27% → 83.17%**, Kappa **0.74 → 0.77**, MCC **0.75 → 0.78**, AUC **0.946 → 0.959**, large-cell recall **0.75 → 0.88**.
- [ ] Can write the Kappa, MCC and specificity formulas on a whiteboard.
- [ ] Can explain *why* the gain is modest and how you'd test significance (McNemar / CV / bootstrap).
- [ ] Can separate the two error causes: imbalance (large cell) vs. visual similarity (adeno/squamous).
- [ ] Have the confusion-matrix heatmap ready to share or sketch.
- [ ] Won't claim anything that isn't in the notebook (no ensemble, no AMP numbers).
