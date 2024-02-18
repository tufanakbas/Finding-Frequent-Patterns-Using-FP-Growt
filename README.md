# Finding Frequent Patterns Using FP-Growth Algorithm

## How To Use
- Clone this repository
- Go into the repository
- Install the necessary libraries
- Run "fp-growth_all.py" first
- Then run "fp-growth_categories.py" first
- After that "frequency.py" and "graph.py" can be run.

## To install the necessary libraries
Run "pip install nltk mlxtend matplotlib seaborn" in terminal

## You can access detailed information about the study via "Data-Science-Report.pdf".
## If you want to access the outputs without running the code, you can find the outputs in the outputs folder.

## Method
On that project, the FP-Growth algorithm is used to find frequent patterns in the 42.000 news dataset. The working principle is realized by creating the FP-Tree, a tree-based structure. In the first step, frequently occurring items in the dataset are identified. Then, the FP-Tree containing these items is created. The FP-Tree represents the frequent patterns in the dataset. When building the FP-Tree, the algorithm compresses the data in such a way that the frequent items are included, and thus efficiently analyzes the data set. Due to this simple structure and efficiency, FP-Growth is used for finding association rules in large data sets.

- **fp-growth_all**: It runs the algorithm on all news data and gives a single output.

- **fp-growth_categories**: It runs the algorithm on all news data, evaluates different categories within themselves and outputs the number of categories.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
---


