# Interview Study Guide — Lung Cancer Prediction with CNN & Transfer Learning

Everything you need to explain this project confidently. Numbers in here come from your actual training run — you can defend every one of them.

---

## 1. The 30-Second Elevator Pitch

> "I built a deep learning model that classifies chest CT scans into three types of lung cancer — adenocarcinoma, large cell carcinoma, squamous cell carcinoma — plus normal. Instead of training a network from scratch, I fine-tuned a ResNet50 that was pretrained on ImageNet, which let me get strong results from only about a thousand medical images. It reached 81.3% accuracy and 0.946 AUC on a held-out test set, and importantly, it never confused a cancerous scan with a normal one — 100% precision and recall on the normal class."

Practice saying this out loud. It's your opener for "tell me about a project."

---

## 2. The Problem and the Data

- **Task:** 4-class image classification of chest CT scans.
- **Classes:** adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal. The long folder names (like `T2_N0_M0_Ib`) are TNM cancer staging codes — Tumor size, lymph Node involvement, Metastasis, and overall stage. You don't predict the stage; it's just part of the label name.
- **Dataset:** the public "Chest CT-Scan images" dataset from Kaggle (~1,000 images), pre-split into train / valid / test folders. Test set = 315 images.
- **Why it's hard:** small dataset (deep networks normally need millions of images), imbalanced classes, and the three cancer types look visually similar to a non-expert.

**Real-world motivation (say this if asked "why this project?"):** Lung cancer is the leading cause of cancer death worldwide, and outcomes depend heavily on early detection. A model like this isn't a doctor replacement — it's a potential *triage/second-reader* tool that could flag suspicious scans for radiologist review.

---

## 3. How It Works — The Pipeline in Plain English

1. **Load images** from folders; the folder name is the label (`ImageFolder`).
2. **Preprocess:** resize everything to 224×224 (what ResNet expects), convert to tensors, normalize with ImageNet channel statistics.
3. **Augment training data only:** random rotation (±10°), horizontal flips, small shifts/zooms, slight brightness/contrast changes. Each epoch the model sees slightly different versions of each image, which acts like having more data and reduces overfitting.
4. **Model:** ResNet50 pretrained on ImageNet. Freeze almost everything; unfreeze the last block (`layer4`) and replace the final classifier with a new head: `Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→4)`.
5. **Train** for 25 epochs with Adam, cross-entropy loss weighted by class frequency, and two learning rates: tiny (1e-5) for `layer4`, larger (1e-4) for the new head. A scheduler halves the learning rate when validation loss stops improving.
6. **Checkpoint:** after every epoch, if validation loss hit a new low, save the model. At the end, load the best checkpoint — not the last one.
7. **Evaluate** on the untouched test set with accuracy, precision/recall/F1, specificity, confusion matrix, Cohen's Kappa, MCC, and ROC curves.

---

## 4. Key Concepts — Explain Like You Built It (Because You Did)

**CNN (Convolutional Neural Network):** a network that learns visual filters. Early layers detect simple things (edges, textures), deeper layers combine them into complex patterns (shapes, structures, tumors). Convolutions share weights across the image, so they need far fewer parameters than fully-connected layers and are naturally good at "the same pattern can appear anywhere in the image."

**Transfer learning:** take a model already trained on a huge dataset (ImageNet, 1.2M images) and adapt it to your task. The intuition: the edge/texture/shape detectors it learned are useful for *any* image task, including CT scans. You only need to teach it the task-specific part. This is *the* reason the project works with ~1,000 images.

**ResNet50 and residual (skip) connections:** a 50-layer CNN. Very deep networks used to be *harder* to train because gradients vanish as they flow back through many layers. ResNet's fix: skip connections that add a layer's input directly to its output (`output = F(x) + x`). Each block only has to learn the *residual* — the change to make — and gradients have a highway back through the network. This is what made 50+ layer networks trainable, and it's the answer to "why ResNet?"

**Freezing vs. fine-tuning:** frozen layers keep their ImageNet weights (no gradient updates). I froze the early layers (generic features — no reason to relearn edges) and fine-tuned only `layer4` (the most task-specific features) plus the new head. Fewer trainable parameters = less overfitting on a small dataset + faster training.

**Discriminative learning rates:** the pretrained `layer4` already has good weights — I nudge it gently (lr 1e-5). The new head starts from random weights — it needs to learn fast (lr 1e-4, 10× higher). One optimizer, two parameter groups, two learning rates.

**Why ImageNet normalization on CT scans:** the pretrained weights were learned on images normalized with ImageNet's mean/std. Feeding inputs with a different distribution would mismatch what the filters expect. CT scans are grayscale but get loaded as 3 identical channels, so the 3-channel model works fine.

**Dropout (0.5):** during training, randomly zero out half the neurons in the head each forward pass. Prevents the network from relying on any single neuron — a strong overfitting defense, needed here because the dataset is small.

**Softmax + cross-entropy:** softmax turns the 4 raw output scores into probabilities summing to 1; cross-entropy loss punishes the model heavily when it assigns low probability to the correct class.

**Class-weighted loss:** the dataset has more adenocarcinoma images than large cell carcinoma. Without correction, the model can lower its loss by favoring common classes. I weighted each class's loss inversely to its frequency, so a mistake on a rare class costs more.

**ReduceLROnPlateau scheduler:** when validation loss stops improving for 3 epochs, halve the learning rate. Big steps early to learn fast, small steps late to settle into a good minimum. I added this after noticing validation loss oscillating in later epochs — a sign the learning rate was too large near the end.

**Checkpointing on validation loss (early-stopping style):** training loss kept falling to 0.07, but validation loss bottomed out around epoch 18 and then wobbled — the model was starting to memorize. By saving only when validation loss improved, I effectively picked the model from its best-generalizing epoch, not the most-memorized one.

---

## 5. Results — Know Your Numbers Cold

| Metric | Value | One-line meaning |
|---|---|---|
| Test accuracy | **81.27%** | % of the 315 test scans classified correctly |
| Weighted AUC-ROC (OvR) | **0.946** | Across thresholds, the model ranks the right class above wrong ones ~95% of the time |
| Cohen's Kappa | **0.74** | Agreement with truth, corrected for lucky guessing (0.61–0.80 = "substantial") |
| MCC | **0.75** | Correlation between predictions and truth; robust to class imbalance (+1 perfect, 0 random) |
| Normal class | **1.00 precision, 1.00 recall** | Zero normal↔cancer confusions on the test set |
| Adenocarcinoma | 0.85 precision, 0.70 recall | Weakest recall — often confused with squamous |
| Large cell | 0.88 precision, 0.75 recall | Smallest class (51 test images) |
| Squamous cell | 0.67 precision, 0.89 recall | Absorbs the adenocarcinoma confusion |
| Specificity per class | 0.83 – 1.00 | How well the model avoids false alarms per class |

**The story in the numbers (memorize this):** the model is *perfect* at cancer vs. no-cancer — every error is between cancer *subtypes*, mainly adenocarcinoma being called squamous cell carcinoma. Clinically, subtype confusion is far less dangerous than missing a cancer. That framing turns your 81% into a strength.

**Why report Kappa/MCC/specificity at all?** Accuracy alone is misleading with imbalanced classes (a model predicting the majority class can score high while being useless). Kappa and MCC correct for chance and imbalance; specificity matters in medicine because false alarms cause unnecessary biopsies and anxiety.

---

## 6. Questions an Interviewer Will Actually Ask (with answers)

### "Why transfer learning instead of training from scratch?"
~1,000 images is nowhere near enough to train a 25M-parameter network from scratch — it would either overfit badly or never learn. A pretrained ResNet50 already knows generic visual features from 1.2M ImageNet images; I only had to adapt its deepest layers to CT scans. It's also dramatically cheaper: I trained on a laptop GPU in a reasonable time.

### "Why ResNet50 and not something else?"
It's a proven, well-understood baseline for transfer learning: deep enough to be expressive, small enough to fine-tune on modest hardware, and its residual connections make it stable to train. Honest add-on: "Given more time I'd benchmark EfficientNet or a Vision Transformer against it — ResNet50 was a deliberate strong baseline, not the end state."

### "What's a residual connection?" (very common follow-up)
A shortcut that adds a block's input directly to its output. It solves vanishing gradients in deep networks because gradients can flow back through the shortcut unimpeded, and each block only learns an adjustment rather than a full transformation.

### "Why did you only unfreeze layer4?"
Early CNN layers learn universal features (edges, textures) that transfer to any domain — retraining them on 1,000 images would only degrade them. `layer4` holds the most abstract, task-specific features, so that's where adaptation to CT imagery pays off. It's a bias/variance tradeoff: more unfrozen layers = more capacity but more overfitting risk on small data.

### "Why two learning rates?"
Pretrained weights are already good — a large learning rate would destroy them (called "catastrophic forgetting"). The new head is random and needs to learn quickly. So: 1e-5 for layer4, 1e-4 for the head.

### "CT scans are grayscale — why a model built for color images?"
The grayscale image is replicated across 3 channels to match the input shape. The pretrained filters still respond to edges, gradients, and textures, which is what matters. I also kept ImageNet normalization because that's the input distribution the pretrained weights expect.

### "How did you handle class imbalance?"
Two ways: class-weighted cross-entropy loss (misclassifying a rare class costs more), and evaluating with imbalance-robust metrics (MCC, Kappa, per-class F1) instead of trusting accuracy. An alternative I could compare against is oversampling with a `WeightedRandomSampler`.

### "Did your model overfit? How do you know?"
Yes, in the later epochs — training loss dropped to ~0.07 while validation loss plateaued around 0.4–0.5 after epoch 18. That growing gap is the signature of memorization. I mitigated it with data augmentation, dropout (0.5), freezing most of the network, and checkpointing on best validation loss so the final model comes from the best-generalizing epoch, not the last one.

### "Which is worse here: a false positive or a false negative?"
A false negative — telling a cancer patient they're healthy delays treatment for a disease where early detection drives survival. A false positive costs a follow-up scan or biopsy: bad, but recoverable. That's why I highlight that my model had zero cancer→normal misclassifications on the test set, and why in deployment I'd tune the decision threshold to favor sensitivity (catching cancer) even at the cost of some specificity.

### "Where does your model fail, and why?"
Almost all errors are adenocarcinoma ↔ squamous cell confusion (adenocarcinoma recall 0.70, squamous precision 0.67). These subtypes can genuinely look similar on CT — even radiologists often rely on biopsy for subtyping. With more data per subtype, higher-resolution inputs, or attention-based models, I'd expect that boundary to improve.

### "How would you improve this project?" (always asked — have a ranked list)
1. **Grad-CAM interpretability** — heatmaps showing *where* the model looks; in medical AI, trust matters as much as accuracy.
2. **A from-scratch CNN baseline** — to quantify exactly how much transfer learning buys.
3. **k-fold cross-validation** — 315 test images is small, so single-split metrics have noise; CV gives confidence intervals.
4. **External validation** — test on scans from a different hospital/scanner; models often drop sharply on new data sources (domain shift).
5. **Modern architectures & ensembling** — EfficientNet, ViT; ensemble for a few extra points.
6. **Test-time augmentation** — average predictions over flipped/rotated copies of each test image.

### "Would you deploy this in a hospital?"
No — and saying so shows maturity. It's trained on ~1,000 images from one public dataset, evaluated on 315 images, with no external validation, no prospective clinical trial, and no regulatory clearance. Realistic role: a research prototype demonstrating that transfer learning can extract strong signal from limited medical imaging data, or a triage aid that *prioritizes* scans for radiologists — never an autonomous diagnostic.

### "What was the hardest part?" (behavioral)
Good honest answer: "Getting good performance from so little data. My first instinct was to train more of the network, but that overfit — the fix was the opposite: freeze more, augment more, regularize more, and select the model by validation loss. It taught me that in small-data regimes, restraint beats capacity." (Adapt to your actual experience.)

### "Walk me through what happens when one image goes through your model."
Resize to 224×224 → normalize → ResNet50 backbone extracts a 2048-dimensional feature vector through ~50 convolutional layers → my head maps it 2048→512 (ReLU, dropout) →512→4 scores → softmax → probabilities → argmax = predicted class.

### Rapid-fire definitions you should be able to give in one sentence:
- **Epoch:** one full pass over the training data (I trained 25).
- **Batch size:** images processed per gradient update (32).
- **Adam:** an optimizer that adapts each parameter's step size using running averages of gradients.
- **Precision:** of everything I predicted as class X, how much really was X.
- **Recall (sensitivity):** of all real class X, how much did I catch.
- **Specificity:** of all *non*-X, how much did I correctly leave alone.
- **F1:** harmonic mean of precision and recall.
- **ROC curve:** true-positive rate vs. false-positive rate across all decision thresholds; AUC is the area under it (1.0 perfect, 0.5 random).
- **Confusion matrix:** table of actual vs. predicted classes; where the errors live.
- **Validation vs. test set:** validation guides training decisions (checkpointing, scheduler); test is touched exactly once, at the end, for the honest final score.

---

## 7. Honest Limitations (own these before they're pointed out)

- Small, single-source public dataset; no external validation → unknown generalization to other scanners/hospitals.
- 315-image test set → metrics have meaningful uncertainty (~±4% on accuracy).
- Labels come with the dataset; I can't verify their clinical ground truth.
- No interpretability yet (Grad-CAM is the top TODO).
- Not a medical device; educational project.

Interviewers respect candidates who state limitations unprompted far more than those who oversell.

---

## 8. Cheat Sheet — Numbers to Have Memorized

- **81.3%** test accuracy · **0.946** weighted AUC · Kappa **0.74** · MCC **0.75**
- **100%/100%** precision/recall on normal (zero cancer↔normal errors)
- **315** test images · **4** classes · **25** epochs · batch **32**
- LRs: **1e-5** (layer4) / **1e-4** (head) · Dropout **0.5**
- ResNet50: **~25M** params, ImageNet-pretrained, only layer4 + head trained
- Best validation loss at **epoch 18** (0.40); training loss ended at 0.07 → overfitting gap → why checkpointing mattered

> Note: these numbers are from the original training run. After re-running the notebook with the new improvements (class-weighted loss, LR scheduler, fixed augmentation), regenerate this cheat sheet with the new results.
