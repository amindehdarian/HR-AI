# Attrition Prediction (PyTorch)
### A Practical Workforce Analytics Use Case Demonstrating Predictive HR Modeling

This project demonstrates how a **predictive attrition model** can be built using **PyTorch** to estimate the likelihood that an employee will voluntarily leave an organization.  
The purpose is to showcase how data science methods can be translated into **actionable workforce insights**, as commonly required in **People Analytics, HR Strategy, and Workforce Planning** roles.

The notebook and documentation are written to illustrate:

- how HR-relevant features can be engineered for predictive modeling  
- how neural networks can be applied to real workforce questions  
- how model outputs can support decision-making in HR and business contexts  
- how such a model could be integrated into a data product or analytics toolkit  

This project is part of a broader portfolio demonstrating applied analytics for HR and workforce use cases.

---

## 1. Business & Analytics Context

### 1.1 What Problem Does Attrition Prediction Solve?

Unplanned turnover creates major operational and financial risks.  
Predicting attrition helps organizations:

- forecast workforce supply gaps  
- identify teams or roles at high risk of churn  
- target retention interventions  
- optimize recruiting pipelines  
- quantify the financial impact of turnover  

Attrition models are foundational in **People Analytics consulting**, **HR digital products**, and **strategic workforce planning**.

### 1.2 Why Use a Neural Network?

While logistic regression and tree-based models are commonly used, a **feed-forward neural network**:

- captures nonlinear relationships between HR attributes  
- handles interactions between features more flexibly  
- aligns with modern AI/ML practices used in HR Tech platforms  
- demonstrates readiness to work with deep learning frameworks used in enterprise HR analytics products  

The goal is not to claim neural networks are always superior, but to show the ability to design models aligned with **current expectations in AI-assisted HR analytics**.

---

## 2. Data and Feature Construction

This section explains how HR-relevant features are structured and how an attrition target variable is created to train the predictive model.  
Understanding this pipeline is essential for **People Analytics, HRIS, or product roles** focused on turning workforce data into insights.

---
### 2.1 Feature Representation

The employee dataset includes `n_samples` individuals and `n_features` numerical workforce indicators, such as:

- age  
- tenure  
- engagement or sentiment measures  
- manager effectiveness scores  
- compensation-related metrics  

These features form a matrix `X` of shape:

\[
X \in \mathbb{R}^{n\_samples \times n\_features}
\]

In code:

```python
X = torch.randn(n_samples, n_features)
```


### 2.2 Latent Relationship Between Features and Attrition

To demonstrate how predictive modeling identifies patterns, the project constructs an underlying relationship between employee characteristics and the probability of leaving the organization. This mirrors real-world scenarios where employee behavior is influenced by multiple workforce factors.

The design reflects common insights applied in People Analytics and HR Strategy:

- lower engagement may increase turnover risk  
- strong managerial support often reduces attrition  
- early-tenure employees may show higher mobility or churn  

These assumptions are encoded through an internal set of weights and a logistic transformation:

```python
weights_true = torch.tensor([...])
logits = X @ weights_true + noise
probs = torch.sigmoid(logits)
```

The transformation from `logits` to `probs` produces a value between **0 and 1** for each employee, which can be interpreted as an estimated attrition probability.

To generate attrition labels (`0` = stay, `1` = leave), we map these probabilities into binary outcomes:

```python
y = torch.bernoulli(probs).long()
```

This produces an attrition outcome for every employee that mirrors the structure of turnover indicators typically stored in HR Information Systems.

- `0` → employee stays  
- `1` → employee leaves  

The two tensors, `X` (employee features) and `y` (attrition labels), form the input–output pair required for building and training the predictive model.

---

### 2.3 Why This Construction Is Valuable for Workforce Analytics

This setup provides a practical and interpretable foundation for understanding how predictive models operate on HR data. By explicitly defining a structured relationship between workforce characteristics and attrition outcomes, the model can learn patterns that resemble those commonly analyzed in People Analytics and HR Strategy functions.

This approach allows us to:

- illustrate how workforce attributes translate into attrition risk  
- examine how model outputs can drive insights for HR Business Partners and managers  
- prototype early-warning systems or retention dashboards  
- explore how AI-driven models can enhance HR decision-making  

The resulting features (`X`) and labels (`y`) can be seamlessly replaced with real employee data in applied settings. This requires standard preprocessing steps such as:

- scaling numerical features  
- encoding categorical variables (job function, region, grade)  
- resolving missing or inconsistent values  
- applying fairness, transparency, and privacy controls  

These steps reflect data-quality workflows typically performed in People Analytics consulting, HR data engineering, and the development of HR Tech products.

---

## 3. Model Architecture

The predictive attrition model is implemented using a simple, interpretable feed-forward neural network.  
This architecture is widely used in HR and workforce analytics when modeling nonlinear relationships or interactions between employee attributes.

---

### 3.1 Input and Output Structure

The model takes as input a feature vector representing an individual employee.  
Each vector contains numerical values describing workforce characteristics such as:

- age  
- tenure  
- engagement or sentiment metrics  
- manager effectiveness indicators  
- compensation-related variables  

Mathematically, each input is a vector:

\[
x \in \mathbb{R}^{n\_features}
\]

The model outputs a **single probability**:

\[
p(\text{attrition} = 1 \mid x) \in [0, 1]
\]

representing the estimated likelihood that the employee will voluntarily leave the organization.

This probability can be interpreted at both individual and group level (e.g., team-level attrition risk, segment-level comparisons).

---

### 3.2 Network Design

The network is defined as follows:

```python
class AttritionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

### 3.3 Training Procedure

The neural network is trained using supervised learning, where each employee is labeled as:

- **1** – voluntarily left the organization  
- **0** – remained employed  

The training objective is to minimize **binary cross-entropy loss**:

\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
\]

where \( y_i \) is the true attrition label and \( \hat{y}_i \) is the predicted probability.

The model is optimized using **Adam**, which is effective for mixed-scale HR data.

**Training configuration:**
- batch size: 32  
- learning rate: 0.001  
- epochs: 50–100  
- validation-based early stopping  

---

### 3.4 Regularization and Model Stability

To ensure the model generalizes well and remains stable over time, several regularization strategies are used:

### **L2 Weight Decay**
Prevents large, unstable weights that could cause erratic predictions.

### **Dropout**
Randomly disables a proportion of neurons during training (typically 20–30%) to reduce overfitting.

### **Feature Standardization**
All numeric inputs are standardized:

\[
x' = \frac{x - \mu}{\sigma}
\]

This improves optimization speed and helps ensure comparable feature influence.

---

### 3.5 Model Evaluation

Attrition datasets are usually imbalanced, so the model is evaluated using metrics beyond accuracy.

### **Primary metrics**
- AUC–ROC  
- Precision–Recall AUC  
- F1-score  
- Balanced accuracy  

### **Secondary diagnostics**
- confusion matrix  
- model calibration curves  
- lift and gain charts  

These capture both predictive performance and practical, business-relevant value.

---

### 3.6 Interpretability and Explainability

For HR models, transparency is essential. The following approaches are used to interpret predictions:

### **Permutation Feature Importance**
Evaluates how much each feature contributes to prediction accuracy.

### **Partial Dependence Plots (PDPs)**
Show how changes in a single feature affect attrition probability.

### **SHAP Values**
Provide individual-level and global explanations, indicating:
- how much each feature increased or decreased predicted attrition risk  
- which factors drive risk across the workforce  

---

### 3.7 Deployment and Practical Use

After validation, the model can be deployed in HR dashboards, workforce planning tools, or retention programs.

### **Applications**
- identifying high-risk employees for targeted retention  
- monitoring team-level attrition hotspots  
- conducting what-if scenario analyses  
- supporting manager decision-making with interpretable risk insights  

### **Ethical and governance safeguards**
- fairness and bias detection  
- transparent documentation  
- restricted access and HR data governance alignment  

These measures ensure responsible use of predictive analytics in workforce decision-making.

---

## 4. Training and Optimization

This section describes how the attrition prediction model is trained, optimized, and validated. The goal is to ensure the model generalizes well to unseen employee data and remains stable for HR decision support.

---

### 4.1 Training Objective

The model is trained to predict whether an employee will voluntarily leave. Given predicted probability \(\hat{y}\) and true label \(y\), the loss function is binary cross-entropy:

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right)
\]

This encourages the network to output probabilities close to the true attrition outcomes.

---

### 4.2 Optimization Strategy

The model is optimized using the Adam optimizer, which:

- adapts learning rates per parameter  
- converges efficiently  
- handles mixed-scale HR datasets  

**Training configuration:**

- batch size: 32  
- learning rate: 0.001  
- epochs: 50–100  
- validation split: 20%  
- early stopping on validation loss  

---

### 4.3 Regularization and Generalization

To reduce overfitting and increase stability, the model incorporates:

#### L2 Weight Decay  
Penalizes large weights to encourage smoother decision boundaries.

#### Dropout  
Randomly disables 20–30% of neurons during training to improve generalization.

#### Feature Standardization  
All numeric inputs are standardized:

\[
x' = \frac{x - \mu}{\sigma}
\]

This improves optimization stability and ensures comparable feature influence.

---

### 4.4 Model Validation

Validation is performed on a held-out dataset that is not used for training.

**Key validation metrics:**

- AUC–ROC  
- Precision–Recall AUC  
- F1-score  
- Balanced accuracy  

These metrics reflect discrimination capability and robustness on imbalanced attrition data.

---

### 4.5 Training Loop (Illustrative Code)

```python
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
```

This loop iteratively updates model parameters to minimize the loss over batches of employee records.

---


## 5. Model Evaluation

This section presents how the attrition model is evaluated after training. The goal is to ensure the model performs reliably, handles class imbalance, and provides actionable signals for HR decision-making.

---

### 5.1 Evaluation Metrics

Employee attrition datasets are typically imbalanced, with far fewer “leavers” than “stayers.”  
Therefore, traditional accuracy is not a meaningful measure.  
Instead, evaluation focuses on metrics that capture performance on minority classes and overall discrimination capability.

**Primary evaluation metrics:**

- **AUC–ROC**  
  Measures the model’s ability to distinguish between leavers and stayers across thresholds.

- **Precision–Recall AUC**  
  More informative when the positive class (attrition) is rare.

- **F1-score**  
  Harmonic mean of precision and recall.

- **Balanced accuracy**  
  Adjusts for the skewed class distribution by averaging sensitivity and specificity.

**Secondary diagnostics:**

- confusion matrix  
- calibration curves  
- lift and gain charts  
- distribution of predicted probabilities across segments (e.g., teams, tenure groups)

---

### 5.2 ROC and Precision–Recall Curves

The **ROC curve** illustrates the trade-off between true-positive and false-positive rates.  
A good model achieves an ROC AUC significantly above 0.70 for HR use cases.

The **Precision–Recall curve** is especially important when only a small percentage of employees leave.  
High precision at reasonable recall ensures that the model identifies meaningful high-risk cases without overwhelming HR teams with false alerts.

Example code to compute curves:

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve

roc_auc = roc_auc_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_pred)
```

### 5.3 Confusion Matrix Analysis

To analyze classification behavior at a chosen threshold (e.g., 0.5), a confusion matrix is used. It summarizes:

- **True Positives (TP):** correctly predicted leavers  
- **False Positives (FP):** predicted leavers who actually stayed  
- **True Negatives (TN):** correctly predicted stayers  
- **False Negatives (FN):** leavers the model failed to identify  

This helps HR stakeholders understand the operational implications of using the model, including:

- the number of likely false alarms  
- how many high-risk employees may be missed  
- how to allocate retention resources effectively  

Example code:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred_labels)
```

### 5.4 Calibration and Probability Quality

Accurate probability calibration is essential because predicted attrition probabilities may be displayed in HR dashboards or used to guide retention actions. Calibration assesses whether the predicted risks match actual observed outcomes. For example:

- employees predicted at 20% risk should leave at roughly 20%  
- employees predicted at 50% risk should leave at roughly 50%  

A calibration curve compares predicted probabilities with observed attrition rates.  
If the model is poorly calibrated, post-training calibration methods such as **Platt scaling** or **isotonic regression** can be applied.

---

### 5.5 Segment-Level Evaluation

Model performance can vary across different parts of the organization.  
Evaluating metrics within workforce segments helps ensure fairness, robustness, and usability. Relevant segments include:

- business unit  
- job family  
- location  
- tenure bands  
- manager or team  

Segment-specific evaluation verifies that:

- no subpopulation is systematically disadvantaged  
- the model performs consistently across diverse employee groups  
- insights remain reliable when used for workforce planning or retention strategies  

---

### 5.6 Summary

This evaluation confirms that the model:

- handles class imbalance effectively  
- produces reliable, interpretable probability outputs  
- identifies high-risk employees accurately  
- generalizes across organizational segments  
- supports evidence-based decision-making in HR contexts  

The next section introduces interpretability methods—such as SHAP values and feature importance—that explain how the model makes its predictions.

---

## 6. Interpretability and Explainability

Predictive models in HR must be transparent and explainable.  
Managers, HR partners, and employees need to understand why a prediction is made, how features influence risk, and whether the model behaves fairly.  
This section outlines the techniques used to interpret individual predictions and global model behavior.

---

### 6.1 Importance of Explainability in HR

Attrition models influence decisions about:

- retention programs  
- workload assessments  
- leadership coaching  
- compensation or development interventions  

Therefore, interpretability ensures:

- **trust:** users understand how predictions are generated  
- **fairness:** patterns can be checked for bias  
- **actionability:** explanations directly guide interventions  
- **compliance:** aligns with AI governance and responsible-analytics principles  

---

### 6.2 Global Feature Importance

Global feature importance identifies which variables most strongly influence attrition predictions across the entire workforce.

A common method is **permutation importance**, which measures how much model error increases when a feature’s values are randomly shuffled.

High-importance examples might include:

- tenure  
- engagement levels  
- manager effectiveness  
- workload indicators  
- compensation competitiveness  

Example code:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_val, y_val, n_repeats=10)
```

### 6.3 Partial Dependence Plots (PDPs)

Partial Dependence Plots illustrate how changes in a single feature affect predicted attrition probability while averaging over all other features. They help HR analysts understand whether a feature has a linear, monotonic, or more complex effect.

Typical insights might include:

- attrition risk decreases steadily with increasing tenure  
- low engagement scores sharply increase predicted risk  
- compensation competitiveness has nonlinear effects (plateaus at high values)  

PDPs provide intuitive, global explanations for feature behaviour across the workforce.

---

### 6.4 SHAP Values (Local and Global Explanations)

SHAP (SHapley Additive exPlanations) values quantify how each feature contributes to an individual prediction.  
They provide both detailed **local** explanations and aggregated **global** insights.

#### Global explanations
- identify top drivers of attrition across the organization  
- reveal nonlinear or interaction effects  
- highlight areas where HR policy may need attention  

#### Local (individual-level) explanations
- clarify why a specific employee has high or low attrition risk  
- show which factors increase risk (positive SHAP values)  
- show which factors reduce risk (negative SHAP values)  
- enable targeted, personalized retention discussions  

Example code:

```python
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_val)
```

### 6.5 Manager-Facing Explanations

To support HR decision-making, predictions must be translated into clear and actionable insights for managers. A typical explanation includes:

- the employee’s predicted attrition probability  
- the top features increasing risk  
- the top features reducing risk  
- suggested follow-up actions or resources  

Example phrasing:

- “Low engagement and a recent manager change are increasing this employee’s attrition risk.”  
- “High performance scores and competitive pay are helping stabilize the risk.”

These explanations make complex model outputs accessible to non-technical stakeholders and ensure the model supports constructive retention discussions.

---

### 6.6 Explainability for Fairness and Bias Testing

Explainability methods also enable fairness evaluation by revealing how the model treats different demographic groups. Key steps include:

- comparing SHAP value distributions across groups  
- checking whether certain features disproportionately influence predictions for specific segments  
- detecting proxy variables that could inadvertently introduce bias  
- confirming consistent model behaviour across tenure, job families, or demographic attributes  

This analysis ensures the model adheres to responsible AI and people analytics ethics standards.

---

### 6.7 Summary

Explainability ensures that the attrition prediction system is:

- transparent and interpretable  
- auditable for fairness  
- actionable for HR and managers  
- aligned with ethical and governance requirements  

The next section will address fairness, bias mitigation, and responsible deployment of predictive models in HR contexts.

---

## 7. Fairness, Bias, and Responsible AI in HR Analytics

Predictive modeling in HR requires high ethical standards. Attrition models influence decisions that affect people’s careers, well-being, and development opportunities.  
This section outlines the principles and procedures used to ensure fair, transparent, and responsible deployment.

---

### 7.1 Why Fairness Matters in Attrition Modeling

HR analytics directly interacts with sensitive human outcomes.  
Unfair or biased models can lead to:

- unequal treatment of employees  
- disadvantaged opportunities for specific groups  
- compliance risks (e.g., GDPR, internal AI policies)  
- erosion of trust in HR systems  

Ensuring fairness protects employees and strengthens the credibility of People Analytics practices.

---

### 7.2 Sources of Bias in Workforce Data

Bias can emerge from multiple stages of the HR data pipeline:

- **Historical bias:** existing inequalities embedded in past decisions (e.g., promotion, pay)  
- **Measurement bias:** inconsistent performance ratings, subjective survey responses  
- **Sampling bias:** low survey participation in certain groups  
- **Proxy features:** variables that correlate strongly with sensitive attributes  

Recognizing these risks is the first step toward mitigation.

---

### 7.3 Bias Detection Methods

Several techniques are applied to assess potential unfairness in the model:

#### **Group-level performance analysis**
- compare precision, recall, and false-positive rates across demographic groups  
- ensure no group is systematically disadvantaged  

#### **SHAP-based fairness checks**
- inspect whether certain features disproportionately drive predictions for specific groups  
- validate that risk drivers are consistent and justified  

#### **Distributional analysis**
- examine whether predicted risk scores differ sharply across protected groups  
- ensure differences are explainable and not driven by spurious correlations  

These steps support robust, transparent fairness assessment.

---

### 7.4 Mitigation Strategies

If bias is detected, several mitigation strategies can be applied depending on severity:

#### **Pre-processing approaches**
- rebalancing the dataset  
- removing or transforming problematic proxy variables  
- imputing missing data in a group-neutral way  

#### **In-processing approaches**
- fairness-aware loss functions  
- group-constrained regularization  
- adversarial debiasing  

#### **Post-processing approaches**
- calibrating risk scores per segment  
- adjusting thresholds for different groups  
- providing contextual explanations to mitigate misinterpretation  

Mitigation should be documented and consistently monitored.

---

### 7.5 Governance, Transparency, and Accountability

Responsible AI in HR requires clear governance structures, including:

- documented model purpose, scope, and limitations  
- regular model audits (performance, fairness, drift monitoring)  
- strict data access controls  
- transparency on how predictions are used in managerial processes  
- human-in-the-loop reviews for sensitive decisions  

These practices ensure alignment with organizational ethics and regulatory expectations.

---

### 7.6 Ethical Use in HR Decision-Making

Even a fair and accurate model must be used responsibly.  
Key principles include:

- predictions support—not replace—managerial judgment  
- no adverse decisions (e.g., termination) based solely on a model  
- model insights should guide discussions, not dictate outcomes  
- employees should not be penalized for algorithmic predictions  
- transparency with employees where appropriate  

The model is designed to inform retention strategy, not automate people decisions.

---

### 7.7 Summary

This section outlined the core principles of responsible AI in HR:

- identifying sources of bias  
- detecting unequal model behaviour  
- applying mitigation strategies  
- building strong governance and accountability structures  
- ensuring ethical, human-centered use  

These safeguards ensure that the attrition model contributes positively to employee experience and organizational fairness.


