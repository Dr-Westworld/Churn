Below is a **toy example** showing, step‐by‐step, exactly how a trained LightGBM ensemble would take each of your three sample rows (with their one‐hot‐encoded & numeric features) and turn them into a numerical “churn probability.” Since we don’t have access to your exact trained model’s internal tree splits and leaf values, we’ll build a **small illustrative ensemble of three very simple “trees”**. We will then show how each of your three cases “flows” through these trees, how the trees’ leaf values add up to a raw score, and finally how that raw score is turned into a probability with the logistic (sigmoid) function.

> **Important:** The numbers below (splits and leaf outputs) are purely **made up for illustration**. In a real LightGBM model, there would be dozens or hundreds of trees, each of varying depth, and each leaf would have its own learned weight. The core idea remains exactly the same.

---

## 1. The Three Sample Rows

First, let’s restate the three rows you gave us (abbreviated to just the columns that matter):

| Case | Contract       | OnlineSecurity | TechSupport | InternetService | tenure | MonthlyCharges | TotalCharges | PaperlessBilling |   PaymentMethod  | y\_true | y\_pred |
| :--: | :------------- | :------------- | :---------- | :-------------- | :----: | :------------: | :----------: | :--------------: | :--------------: | :-----: | :-----: |
|   1  | Month-to-month | No             | No          | DSL             |    1   |      24.6      |     24.6     |        No        | Electronic check |    1    |    0    |
|   2  | Month-to-month | No             | No          | DSL             |    1   |      24.2      |     24.2     |        No        |   Mailed check   |    0    |    0    |
|   3  | Month-to-month | No             | No          | DSL             |    1   |      43.95     |     43.95    |        Yes       | Electronic check |    1    |    1    |

* **y\_true** is the actual churn label (0 = No, 1 = Yes).
* **y\_pred** is what the LightGBM model predicted (0 or 1).

Notice that all three rows share most of the same values—only two fields differ among them:

1. **MonthlyCharges / TotalCharges** (Case 1→24.6, Case 2→24.2, Case 3→43.95)
2. **PaperlessBilling** (Case 1 & 2 → No, Case 3 → Yes)
3. **PaymentMethod** (Case 1 & 3 → Electronic check, Case 2 → Mailed check)

Everything else (Contract, OnlineSecurity, TechSupport, InternetService, tenure, etc.) is identical across the three.

---

## 2. A Tiny “Toy” LightGBM Ensemble of 3 Trees

Let’s build a small ensemble of **three** binary classification trees.  Each tree $f_m$ outputs a real number (its “leaf value”) once you drop your row down its splits. Our final raw score is:

$$
F(x) \;=\; f_1(x)\;+\;f_2(x)\;+\;f_3(x).
$$

We then turn that raw score $F(x)$ into a probability

$$
\hat p(x) \;=\; \sigma\bigl(F(x)\bigr)
\;=\; \frac{1}{1 + e^{-\,F(x)}}.
$$

If $\hat p(x)\ge0.5$, we predict churn = 1; otherwise churn = 0.

### 2.1 Tree 1: Splits on “MonthlyCharges > 30?”

```
                 [Is MonthlyCharges > 30?]
                     /             \
                   Yes             No
                  /                 \
           leaf value = +0.8      leaf value = –0.2
```

* If **MonthlyCharges > 30**, Tree 1 outputs **$+0.8$**.
* Otherwise (MonthlyCharges ≤ 30), Tree 1 outputs **$-0.2$**.

---

### 2.2 Tree 2: Splits on “PaperlessBilling = Yes?” then on “tenure < 6?”

```
                    [PaperlessBilling = Yes?]
                       /             \
                     Yes             No
                    /                 \
        [tenure < 6?]           leaf value = +0.1
         /          \
       Yes          No
      /              \
 leaf = +0.6      leaf = –0.3
```

* If **PaperlessBilling = No**, Tree 2 immediately outputs **$+0.1$** (because it went right).
* If **PaperlessBilling = Yes**, it checks “tenure < 6?”:

  * If tenure < 6, leaf = **$+0.6$**.
  * Else (tenure ≥ 6), leaf = **$-0.3$**.

---

### 2.3 Tree 3: Splits on “PaymentMethod = Electronic check?”

```
                   [PaymentMethod = Electronic check?]
                          /               \
                        Yes               No
                        /                  \
                 leaf = +0.5           leaf = –0.1
```

* If **PaymentMethod** is “Electronic check,” Tree 3 outputs **$+0.5$**.
* Otherwise (e.g. “Mailed check,” “Credit card (automatic),” etc.), Tree 3 outputs **$-0.1$**.

---

### 2.4 How These Trees Were Hypothetically Trained

* In a real LightGBM run, these “leaf values” (+0.8, –0.2, +0.6, –0.3, +0.1, +0.5, –0.1) would be automatically chosen to **minimize logistic loss** on your training data.
* We are simply **making them up** to illustrate how, once trees are built, you do **no further learning** at inference time—just lookup and sum.

---

## 3. One-Hot & Numeric Input Vector

Before we feed a row into these trees, note that all categorical fields (Contract, OnlineSecurity, etc.) have already been turned into 0/1 dummy columns.

However, in our toy trees, we only actually **look at**:

* “MonthlyCharges” (numeric)
* “PaperlessBilling” (categorical, but as a 0/1 dummy: 1 if Yes, 0 if No)
* “tenure” (numeric)
* “PaymentMethod” (dummy: 1 if Electronic check, 0 otherwise)

All the other columns (Contract, OnlineSecurity, etc.) aren’t used by these toy trees at all, so you can ignore them for inference. (In a real model, many more splits on “Contract,” “TechSupport,” etc. would exist.)

Let’s write down, for each of the three cases, the values of exactly the features our three trees need:

* **Case 1**:

  * MonthlyCharges = 24.6
  * PaperlessBilling = “No”  → dummy = 0
  * tenure = 1
  * PaymentMethod = “Electronic check” → dummy = 1

* **Case 2**:

  * MonthlyCharges = 24.2
  * PaperlessBilling = “No”  → dummy = 0
  * tenure = 1
  * PaymentMethod = “Mailed check” → dummy = 0

* **Case 3**:

  * MonthlyCharges = 43.95
  * PaperlessBilling = “Yes” → dummy = 1
  * tenure = 1
  * PaymentMethod = “Electronic check” → dummy = 1

---

## 4. Step-by-Step Inference for Each Case

### 4.1 Case 1 (y\_true = 1, y\_pred = 0)

| Feature                  | Value                |
| ------------------------ | -------------------- |
| MonthlyCharges           | 24.6 (≤ 30)          |
| PaperlessBilling (dummy) | 0 (No)               |
| tenure                   | 1 (< 6)              |
| PaymentMethod (dummy)    | 1 (Electronic check) |

1. **Tree 1**: Check `MonthlyCharges > 30?`

   * 24.6 ≤ 30 → “No” branch → leaf value = **–0.2**.

2. **Tree 2**: Check `PaperlessBilling = Yes?`

   * It’s 0 (No) → go to “No” branch → leaf value = **+0.1**.

3. **Tree 3**: Check `PaymentMethod = Electronic check?`

   * It’s 1 (Yes) → “Yes” branch → leaf value = **+0.5**.

**Sum the three leaf‐values** to get the raw score $F(x)$:

$$
F_{\text{Case1}} \;=\; (-0.2) \;+\; (0.1)\;+\;(0.5) \;=\; +0.4.
$$

Now convert to a probability via the sigmoid:

$$
\hat p_{\text{Case1}} \;=\; \frac{1}{1 + e^{-\,0.4}}
\;=\; \frac{1}{1 + e^{-0.4}}
\;\approx\; 0.5987\quad(\text{about } 59.9\%).
$$

* Because $0.5987 > 0.5$, we would predict **churn = 1**.
* However, your actual model predicted **y\_pred = 0**. That means either (a) the real LightGBM had somewhat different leaf‐values (perhaps the sum was < 0, yielding a probability < 0.5), or (b) the model threshold was set above 0.5. In our toy example, we see a predicted 59.9 % churn, but in the real model those three trees’ leaves might not have summed to +0.4—maybe they summed to a small negative number instead.

---

### 4.2 Case 2 (y\_true = 0, y\_pred = 0)

| Feature          | Value       |
| ---------------- | ----------- |
| MonthlyCharges   | 24.2 (≤ 30) |
| PaperlessBilling | 0 (No)      |
| tenure           | 1 (< 6)     |
| PaymentMethod    | 0 (Mailed)  |

1. **Tree 1**: `24.2 ≤ 30` → leaf = **–0.2**.
2. **Tree 2**: `PaperlessBilling = No` → leaf = **+0.1**.
3. **Tree 3**: `PaymentMethod = Electronic check?` is **0** → leaf = **–0.1**.

Sum:

$$
F_{\text{Case2}} = (-0.2) \;+\; (0.1)\;+\;(-0.1) = -0.2.
$$

Sigmoid:

$$
\hat p_{\text{Case2}} = \frac{1}{1 + e^{-(-0.2)}}
                   = \frac{1}{1 + e^{\,0.2}}
                   \approx 0.4502 \quad(\text{about }45.0\%).
$$

* Since $0.4502 < 0.5$, the model predicts **churn = 0**.
* That matches **y\_pred = 0**. Good.

---

### 4.3 Case 3 (y\_true = 1, y\_pred = 1)

| Feature          | Value          |
| ---------------- | -------------- |
| MonthlyCharges   | 43.95 (> 30)   |
| PaperlessBilling | 1 (Yes)        |
| tenure           | 1 (< 6)        |
| PaymentMethod    | 1 (Electronic) |

1. **Tree 1**: `43.95 > 30` → leaf = **+0.8**.
2. **Tree 2**: `PaperlessBilling = Yes`, then check `tenure < 6?`

   * tenure = 1 < 6 → leaf = **+0.6**.
3. **Tree 3**: `PaymentMethod = Electronic check` → leaf = **+0.5**.

Sum:

$$
F_{\text{Case3}} = (+0.8) \;+\; (+0.6)\;+\;(+0.5) = +1.9.
$$

Sigmoid:

$$
\hat p_{\text{Case3}}
= \frac{1}{1 + e^{-1.9}}
\approx 0.8699 \quad(\text{about }87.0\%).
$$

* $0.8699 > 0.5$, so the model predicts **churn = 1**.
* That matches **y\_pred = 1**.

---

## 5. Putting It All Together

| Case | Tree 1 Output | Tree 2 Output | Tree 3 Output | Raw Score $F(x)$ | Probability $\hat p$ | Final $\; \hat p \ge 0.5\;?$ | y\_pred |
| :--: | :-----------: | :-----------: | :-----------: | :--------------: | :------------------: | :--------------------------: | :-----: |
|   1  |      –0.2     |      +0.1     |      +0.5     |      $+0.4$      |        0.5987        |       Yes (predict = 1)      |   0\*   |
|   2  |      –0.2     |      +0.1     |      –0.1     |      $-0.2$      |        0.4502        |       No (predict = 0)       |    0    |
|   3  |      +0.8     |      +0.6     |      +0.5     |      $+1.9$      |        0.8699        |       Yes (predict = 1)      |    1    |

> \* In our toy‐tree example, Case 1 would produce $\hat p\approx0.5987$ → a predicted churn of 1. But your real LightGBM predicted 0 for Case 1—so the real trees’ leaf‐values or splits must differ slightly. We included Case 1 here mainly to show the **calculation process** step by step. In practice, you would use the **exact** learned leaf outputs from your trained model.

---

## 6. Why This Demonstrates “How LightGBM Calculates Mathematically”

1. **One‐Hot Encoding** turns each categorical field into 0/1 flags (so the model can ask questions like “Is `PaymentMethod_Electronic check = 1`?” as a numeric split).
2. Each **tree** $f_m$ is just a sequence of “if (feature ≤ threshold) go left, else go right,” ending in a constant output (called the **leaf value**, $\gamma_{m,j}$). This value is learned during training to minimize logistic loss (binary cross‐entropy) on the training set.
3. To predict, you “drop your input row down each tree” (checking 0/1 or numeric thresholds) and read off the leaf’s constant value.
4. You **sum** all $M$ leaf‐values to get a raw log‐odds score $F(x)$.
5. You pass $F(x)$ through the **sigmoid** $\sigma(z)=1/(1+e^{-z})$ to get a probability $\hat p(x)\in(0,1)$.
6. If $\hat p\ge 0.5$, LightGBM returns “1” (predict churn); otherwise it returns “0” (predict no churn).
7. **That is the entire mathematical process.**

In a real‐world LightGBM model you might have $M=100$ trees of depth 6 or 8; you’d do exactly the same thing—just more summations:

$$
    F(x)\;=\;\sum_{m=1}^{100} \gamma_{m,j_m(x)},
    \quad \hat p=\sigma(F(x)).
$$

Nothing mystical happens beyond that.

---

## 7. Why Case 1 Might “Underperform” in the Real Model

Your real LightGBM returned **y\_pred 0** for Case 1, even though our toy leaf‐value choices would have predicted 1. That can only happen if, in the **actual learned trees**:

* The split thresholds or leaf weights are slightly different. For instance, the real model might have learned:

  * Tree 1 leaf value for “MonthlyCharges ≤ 30” = –0.3 (instead of –0.2),
  * or Tree 3 might have leaf = +0.3 rather than +0.5,
  * or there might be additional trees that collectively push $F(x)$ down below 0.
* The model may have used a **threshold** other than 0.5 to decide “churn” vs. “no‐churn.” (Sometimes practitioners pick 0.4 or 0.6 to optimize precision or recall specifically.)

What is crucial is that **no matter how many trees are in the ensemble**, each tree simply adds a constant value based on which leaf you end up in. Summing those constants → raw score → sigmoid → probability → final 0/1 decision.

---

### In Summary

* **Step 1:** One‐hot encode every categorical feature so all inputs are numeric.
* **Step 2:** Feed the numeric vector $x$ into each LightGBM tree $f_m$. Each tree routes $x$ to a leaf and returns that leaf’s fixed “leaf value.”
* **Step 3:** Sum all leaf outputs to get a single raw score $F(x)$.
* **Step 4:** Compute

  $$
    \hat p(x) \;=\; \frac{1}{1 + e^{-\,F(x)}},
  $$

  the probability of churn.
* **Step 5:** If $\hat p(x)\ge 0.5$, predict churn (1); otherwise, predict no churn (0).

For each of your three specific rows, we have shown how *a small three‐tree ensemble* would generate a raw score and convert it to a probability. In your real LightGBM model, you would do exactly the same—just with many more trees and different learned leaf‐values.
