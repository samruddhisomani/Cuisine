# Agenda
- Context
- Question
- Data
- Exploratory
- Design Choices
- One Shot Models
- Pipeline Models
- Takeaways

# Context
- What is Yummly? (Screenshot)

# Question
- Given just the ingredients, can we predict which cuisine

# Data
- Screenshot of JSON file
- 30000 training rows
- How many test rows

# Exploratory
- Most Common Cuisine/Most Common Ingredient
  - Unbalanced Classes
- Word Clouds of Top Cuisines

# Design Choices
- Whether to tokenize by word?
- Whether to balance classes?
- How to DTM?
  - Five Minimum Documents?
  - TF IDF? 

# One Shot Models (Do these models miss the same things?)
- Naive Bayes: 78% Accuracy
- Unregularized Logistic Regression: 78%
- Regularized Logistic Regression? 78%
- Support Vector Machine? 78%
- Random Forest? 42%-->Takes forever
- XGBoost? 72%-->Takes forever
- Neural Network?

# Pipeline Models
- DecisionTree--> Cooperative Ensemble
- PCA/LDA-->Logistic Regression (Sparse is good. PCA destroys sparse.)
- RandomForestEmbedding --> Regularized Logistic Regression (Will probably skip. Random forests are no fun).
