# Decision Tree: Classification and Regression Guide

This guide provides a comprehensive, step-by-step walkthrough of Decision Trees, covering both **Classification** (predicting categorical outcomes) and **Regression** (predicting continuous numerical outcomes).

---

## 📖 Table of Contents

1. [Introduction to Decision Trees](#1-introduction-to-decision-trees)
2. [Decision Tree Terminology](#2-decision-tree-terminology)
3. [Classification Trees (Categorical Data)](#3-classification-trees-categorical-data)
   - [Core Definitions & Formulas](#core-definitions--formulas)
   - [Step-by-Step Classification Example](#step-by-step-classification-example)
4. [Regression Trees (Continuous Data)](#4-regression-trees-continuous-data)
   - [Core Definitions & Formulas](#core-definitions--formulas-1)
   - [Step-by-Step Regression Example](#step-by-step-regression-example)
5. [Appendix: Logarithm Calculation Reference](#appendix-logarithm-calculation-reference)

---

## 1. Introduction to Decision Trees

A **Decision Tree** is a supervised machine learning algorithm (a specialized type of probability tree) designed to aid in making decisions about a particular process. It breaks down complex problems into branches, with each branch representing a potential outcome.

---

## 2. Decision Tree Terminology

Understanding the structure of a tree requires knowing its components:

- **Root Node (Level 0):** The starting point of the tree containing the entire dataset.
- **Parent Node:** A node that splits into further sub-nodes.
- **Child Node:** A node resulting from the split of a parent node.
- **Siblings:** Nodes that share the exact same parent.
- **Sub-tree:** A section or branch of the larger tree.
- **Leaf Node:** Terminal nodes that do not split further; these represent the final predicted class or value.

---

## 3. Classification Trees (Categorical Data)

Classification trees predict categories (e.g., "Yes" or "No").

### Core Definitions & Formulas

1.  **Entropy ($E$)**
    Entropy is a measure of disorder, impurity, or randomness in the given dataset. In a decision tree, messy data are split based on values of the feature vector associated with each data point to minimize this disorder.
    $$E(S) = \sum_{i=1}^{c} - p_i \log_2 p_i$$
    _(Where $c$ is the number of classes, and $p_i$ is the probability of class $i$)_

2.  **Information Gain ($Gain$)**
    Information Gain is the difference between the entropy of the parent node and the weighted average entropy of the child nodes. The attribute with the highest Information Gain is chosen as the splitting node.
    $$Gain(T, X) = Entropy(T) - Entropy(T, X)$$

3.  **Gini Index ($I_G$)**
    An alternative to Entropy. It is a powerful measure of the randomness or impurity in the values of a dataset. It aims to decrease impurities from the root node.
    $$I_G = 1 - \sum_{j=1}^{c} p_j^2$$

### Step-by-Step Classification Example

**Goal:** Determine whether to "Wear a Jacket" based on Outlook, Temperature, and Routine.

| Days | Outlook | Temperature | Routine | Wear Jacket? (Target) |
| :--- | :------ | :---------- | :------ | :-------------------- |
| 1    | Sunny   | Cold        | Indoor  | No                    |
| 2    | Sunny   | Warm        | Outdoor | No                    |
| 3    | Cloudy  | Warm        | Indoor  | No                    |
| 4    | Sunny   | Warm        | Indoor  | No                    |
| 5    | Cloudy  | Cold        | Indoor  | Yes                   |
| 6    | Cloudy  | Cold        | Outdoor | Yes                   |
| 7    | Sunny   | Cold        | Outdoor | Yes                   |

#### Step 1: Calculate the Root Entropy (Entropy Before Partition)

Calculate the target variable's overall entropy. Out of 7 days, 4 are **No**, and 3 are **Yes**.
$$E(S) = - \left(\frac{4}{7} \log_2 \frac{4}{7}\right) + \left(- \frac{3}{7} \log_2 \frac{3}{7}\right)$$
$$E(S) = (-0.57 \log_2 0.57) + (-0.43 \log_2 0.43)$$
$$E(S) \approx 0.985$$

#### Step 2: Calculate Conditional Entropy for Each Feature

Evaluate how pure the subsets become if we split the data by each attribute.

**For Outlook:**

- **Sunny (4 instances):** 3 No, 1 Yes. $\rightarrow E(\text{Sunny}) = 0.812$
- **Cloudy (3 instances):** 1 No, 2 Yes. $\rightarrow E(\text{Cloudy}) = 0.918$

**For Temperature:**

- **Cold (4 instances):** 1 No, 3 Yes. $\rightarrow E(\text{Cold}) = 0.812$
- **Warm (3 instances):** 3 No, 0 Yes. $\rightarrow E(\text{Warm}) = 0.00$ (Pure node!)

**For Routine:**

- **Indoor (4 instances):** 3 No, 1 Yes. $\rightarrow E(\text{Indoor}) = 0.812$
- **Outdoor (3 instances):** 1 No, 2 Yes. $\rightarrow E(\text{Outdoor}) = 0.918$

#### Step 3: Calculate Information Gain for Each Feature

Subtract the weighted average conditional entropy from the Root Entropy.

- **Gain(Outlook):** $0.985 - [ (4/7 \times 0.812) + (3/7 \times 0.918) ] = \mathbf{0.127}$
- **Gain(Temperature):** $0.985 - [ (4/7 \times 0.812) + (3/7 \times 0) ] = \mathbf{0.520}$
- **Gain(Routine):** $0.985 - [ (4/7 \times 0.812) + (3/7 \times 0.918) ] = \mathbf{0.127}$

#### Step 4: Select the Root Node

Compare the Information Gains. Because **Temperature (0.520)** has the highest Information Gain, it becomes the Root Node.

#### Step 5: Repeat for Subsets

The "Warm" branch has an entropy of 0 (Pure: All "No"), so it becomes a Leaf Node. The "Cold" branch has an entropy of 0.812, meaning the data is still messy. You must repeat Steps 2-4 exclusively on the "Cold" data subset to find the next best split (e.g., splitting by Outlook or Routine next).

---

## 4. Regression Trees (Continuous Data)

When the target variable is a continuous number (e.g., Salary) instead of a category, we cannot use Entropy. Instead, we use **Variance Reduction**.

### Core Definitions & Formulas

1.  **Variance**
    Calculates how spread out the numbers in a node are from their mean prediction.
    $$\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - Y^{\text{pred}})^2$$
    _(Where $Y_i$ is the actual value, and $Y^{\text{pred}}$ is the mean average of the node)_

2.  **Variance Reduction**
    The equivalent of Information Gain for continuous data. It measures how much the total variance decreases after a split.
    $$\text{Variance Reduction} = \text{Var(root)} - \sum w_i \cdot \text{Var(Child}_i)$$

### Step-by-Step Regression Example

**Goal:** Predict continuous "Salary" based on "Experience" and "Gap".

| Experience | Gap | Salary (Target) |
| :--------- | :-- | :-------------- |
| 2          | Yes | 40              |
| 2.5        | Yes | 42              |
| 3          | No  | 52              |
| 4          | No  | 60              |
| 5          | Yes | 56              |

#### Step 1: Calculate the Root Variance

Find the average Salary for the whole dataset: $(40+42+52+60+56) / 5 = \mathbf{50}$.
Calculate variance from this mean:
$$\text{Variance}_{\text{Root}} = \frac{1}{5} [ (40-50)^2 + (42-50)^2 + (52-50)^2 + (60-50)^2 + (56-50)^2 ]$$
$$\text{Variance}_{\text{Root}} = \frac{1}{5} [ 100 + 64 + 4 + 100 + 36 ] = \mathbf{60.8}$$

#### Step 2: Test a Split and Calculate Child Variances

Let's test a split where **Experience $\le 2.5$**.
This creates two child nodes:

- **Child 1 ($\le 2.5$):** Salaries are [40, 42].
- **Child 2 ($> 2.5$):** Salaries are [52, 60, 56].

Next, you calculate the specific Variance for Child 1 and Child 2 using the same variance formula applied to their local averages.

#### Step 3: Calculate Variance Reduction

Use the formula to see how much variance was removed:
$$\text{Variance Reduction} = 60.8 - \left[ \left(\frac{2}{5} \times \text{Var(Child 1)}\right) + \left(\frac{3}{5} \times \text{Var(Child 2)}\right) \right]$$

#### Step 4: Select the Best Split

Repeat Steps 2 & 3 for all possible numerical thresholds (e.g., $\le 3$, $\le 4$) and categorical variables. The split that yields the **highest Variance Reduction** is chosen as the node.

---

## Appendix: Logarithm Calculation Reference

Calculating base-2 logs by hand/calculator can be tricky. Use these conversions:

**Integer Example:**
$$\log_2 16 = \log_2 (2^4) = 4 \log_2 2 = 4 \times 1 = 4$$
_Alternative Calculator Method:_ $\frac{\log 16}{\log 2} = 4$

**Fraction Example:**
$$\log_2 \left(\frac{1}{16}\right) = \frac{\log(1/16)}{\log 2} = \frac{\log 1 - \log 16}{\log 2} = \frac{0 - \log(2^4)}{\log 2} = -4$$
