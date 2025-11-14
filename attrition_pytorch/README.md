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

Each row corresponds to one employee; each column corresponds to one HR-related predictor.

---

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
The transformation from `logits` to `probs` produces a value between **0 and 1** for each employee, which can be interpreted as an estimated attrition probability.

To generate attrition labels (`0` = stay, `1` = leave), we map these probabilities into binary outcomes:

```python
y = torch.bernoulli(probs).long()

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

## 3.3 Training Procedure

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

## 3.4 Regularization and Model Stability

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

## 3.5 Model Evaluation

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

## 3.6 Interpretability and Explainability

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

## 3.7 Deployment and Practical Use

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

