Below is a step-by-step, mathematical view of exactly how LightGBM takes your one-hot-encoded (and numeric) features and turns them into a final “churn probability” (and then a binary churn/no‐churn decision). We’ll assume you have already:

1. One‐hot encoded every categorical variable so that your input feature vector $x$ is entirely numerical (0s and 1s for the dummies, plus your original numeric columns).
2. Trained a LightGBM binary‐classification model on that data, minimizing the logistic (binary cross‐entropy) loss.

---

## 1. Gradient Boosting Recap: The Additive Model

LightGBM builds an ensemble of $M$ decision trees, where each tree is an additive “correction” to the previous ones. Concretely, the final model’s raw score for an input $x$ is:

$$
    F_M(x) \;=\; \sum_{m=1}^{M} f_m(x),
$$

where:

* $m$ indexes each tree (from 1 to $M$),
* $f_m(x)$ is the prediction (a real‐valued number) of the $m$-th tree when you drop $x$ down that tree.

At iteration $m$, LightGBM has already learned trees $f_1, f_2, \dots, f_{m-1}$. It computes a vector of “pseudo-residuals” from the current ensemble’s output $F_{m-1}(x_i)$ on each training example $x_i$. Then it fits a new tree $f_m(x)$ to approximate those residuals. Summing all $M$ trees gives you the final raw score $F_M(x)$.

---

## 2. From Raw Score to Probability

Because you are doing a binary classification (churn vs. no‐churn), LightGBM uses a logistic (sigmoid) link function at the end. The raw score $F_M(x)$ is on the log‐odds scale. To convert it into a churn‐probability between 0 and 1, you pass it through the sigmoid function:

$$
    \hat{p}(x) \;=\; \sigma\!\bigl(F_M(x)\bigr)
    \;=\; \frac{1}{1 + \exp\bigl(-\,F_M(x)\bigr)}.
$$

So:

1. Each tree $f_m(x)$ outputs a real number (often called the “leaf‐value” or “learning rate‐scaled leaf output”).
2. You sum them to get $F_M(x)$.
3. You apply $\sigma(\cdot)$ to obtain the predicted churn probability $\hat{p}(x)\in(0,1)$.

If you want a final binary decision (“this customer will churn” vs. “will not churn”), you compare $\hat{p}(x)$ to a threshold (e.g. 0.5). So:

$$
    \text{predict “churn” if } \hat{p}(x) \ge 0.5;\quad
    \text{otherwise “no‐churn.”}
$$

Many pipelines adjust that threshold (e.g. 0.30 or 0.40) if they prefer higher recall or precision, but the default is $0.5$.

---

## 3. Diving Deeper: What Is Each Tree’s Output $f_m(x)$?

Each tree $f_m$ is a **decision tree of depth up to some max**, built on the one-hot-encoded (and numeric) features. Mathematically, you can write:

1. Partition the feature space into $J_m$ disjoint “leaf regions” $\{R_{m,1}, R_{m,2}, \dots, R_{m,J_m}\}$.
2. For each leaf region $R_{m, j}$, there is a constant value $\gamma_{m,j}$ that the tree predicts whenever $x \in R_{m,j}$.

Hence,

$$
    f_m(x) \;=\; \sum_{j=1}^{J_m} \gamma_{m,j} \cdot \mathbf{1}\bigl[x \in R_{m,j}\bigr],
$$

where

* $\mathbf{1}[x \in R_{m,j}] = 1$ if $x$ falls into leaf $j$ of tree $m$, and 0 otherwise.
* $\gamma_{m,j}$ is the learned value for that leaf.

During training, at each boosting iteration $m$, LightGBM chooses both *how* to split and *what* leaf‐values $\gamma_{m,j}$ should be, to minimize the total logistic loss on the training set (plus any regularization terms). But at prediction time, you do not re‐learn anything; you simply route $x$ down each tree $f_m$ until you hit a leaf, read off $\gamma_{m,j}$, then sum those $\gamma_{m,j}$ across all $m = 1 \dots M$.

---

## 4. Example: One Tree’s Decisions Over One‐Hot Features

Suppose you have a one-hot-encoded feature vector $x$ containing:

* `Contract_Month-to-month` = 0 or 1
* `Contract_One year` = 0 or 1
* `Contract_Two year` = 0 or 1
* `InternetService_DSL` = 0 or 1
* `InternetService_Fiber` = 0 or 1
* `InternetService_No` = 0 or 1
* `OnlineSecurity_Yes` = 0 or 1
* … plus your numeric columns `tenure`, `MonthlyCharges`, `TotalCharges`, etc.

A small example tree $f_3(x)$ might look like:

```
               [Split on Contract_Month-to-month?]
                   /                       \
                 Yes                        No
                /   \                      /   \
      [leaf value = γ3,1]         [Split on tenure < 12?]
                                      /             \
                                 Yes                 No
                                /   \               /   \
                  [leaf: γ3,2]   [leaf: γ3,3]   [leaf: γ3,4] [leaf: γ3,5]
```

* If `Contract_Month-to-month = 1`, then $f_3(x) = \gamma_{3,1}$.
* If `Contract_Month-to-month = 0` *and* `tenure < 12`, then $f_3(x) = \gamma_{3,2}$.
* If `Contract_Month-to-month = 0, tenure ≥ 12, and so on…`, you land in whatever leaf.

You do that for *each* tree $f_1, f_2, \dots, f_M$. Each tree spits out a real number. Summing them gives you $F_M(x)$.

---

## 5. Putting It All Together: The Prediction Path

1. **One-Hot Encode** your new customer’s raw features → numerical vector $x$.
2. **Pass $x$ through each of the $M$ trees** in your trained LightGBM ensemble:

   * For $m = 1$ to $M$:

     * Traverse tree $m$ using the binary & numeric features in $x$.
     * Find the leaf index $j$ (a single integer).
     * Read off the leaf value $\gamma_{m,j}$.
     * That is $f_m(x) = \gamma_{m,j}$.
3. **Sum** all those leaf‐values:

   $$
       F_M(x) = \sum_{m=1}^{M} \gamma_{m,\,\text{leaf}(x; m)}.
   $$
4. **Apply the sigmoid** to get a probability:

   $$
       \hat{p}(x) = \frac{1}{1 + \exp\bigl(-\,F_M(x)\bigr)}.
   $$

   * If $\hat{p}(x) \ge 0.5$, predict “Churn.” Otherwise, predict “No Churn.”

In other words, LightGBM is learning **an additive set of decision rules**, each rule giving a real-valued “boost” to the log-odds of churn. Adding up all those boosts yields a final logit $F_M(x)$. The logistic function then gives you a probability.

---

## 6. Why One-Hot Encoding Matters

* Each one-hot column (e.g. `Contract_Month-to-month`) is a binary feature. In a split like `[Split on Contract_Month-to-month?]`, the tree checks “is that one‐hot bit = 1 or 0?”
* If you instead left `Contract` as a single string, LightGBM would not be able to evaluate “Is `Contract = Month-to-month`?” in a numeric split. One-hot encoding turns every possible category into a numeric 0/1 flag, which fits cleanly into LightGBM’s decision‐tree splitting logic.

---

## 7. Mathematical Summary

1. **Input**:

   $$
   x = [\,\underbrace{0,1,0}_{\text{Contract dummies}},\,\underbrace{1,0,0}_{\text{Internet dummies}},\,1_{\text{OnlineSecurity}},\,0_{\text{OnlineBackup}},\,\dots,\,\text{tenure},\,\text{MonthlyCharges},\,\text{TotalCharges}\,]^\mathsf{T}.
   $$
2. **Each tree** $f_m$ partitions the one-hot & numeric space and returns a constant $\gamma_{m,\,j}$ for whichever leaf region $x$ falls into.

   $$
       f_m(x) \;=\; \gamma_{m,\,j_m(x)} \quad\text{where $j_m(x)$ is the leaf index for $x$ in tree $m$.}
   $$
3. **Aggregate** the $M$ trees:

   $$
       F_M(x) \;=\; \sum_{m=1}^{M} f_m(x) \;=\; \sum_{m=1}^{M} \gamma_{m,\,j_m(x)}.
   $$
4. **Sigmoid** to get probability:

   $$
       \hat{p}(x) \;=\; \frac{1}{1 + \exp\bigl(-F_M(x)\bigr)}.
   $$
5. **Threshold** (e.g. 0.5) yields a final “churn” (1) vs. “no churn” (0) decision.

---

### TL;DR

* **One-hot encoding** converts each categorical value into a 0/1 feature so that LightGBM can split on it.
* **Each decision tree** learns a partition of the numeric+one-hot space and assigns a constant “boost” (leaf value) $\gamma_{m,j}$ to that region.
* **Summing** all $M$ tree outputs yields a raw log-odds score $F_M(x)$.
* **Applying sigmoid** to $F_M(x)$ transforms it into $\hat p(x) = P(\text{churn} \,\vert\, x)$.
* If $\hat p(x)\ge0.5$, you predict churn; otherwise, you predict no‐churn.

That is exactly how LightGBM “mathematically” computes a churn probability from your one-hot encoded dataset.
