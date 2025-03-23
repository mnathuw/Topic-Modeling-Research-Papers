# 📚 Topic-Modeling-Research-Papers

Welcome to the **Topic Modeling for Research Papers** project! This project was developed as part of **CP421 Data Mining course** to analyze and categorize research papers into topics using unsupervised learning models. The goal is to extract meaningful themes from scholarly articles, helping researchers identify trends and key topics in their field.

## 🎥 Demo
[Click here to watch the video demo](#)

## 🎯 Features
- **Text Preprocessing**: Tokenization, stopword removal, stemming/lemmatization.
- **TF-IDF Representation**: Converts text into numerical vectors for analysis.
- **Topic Modeling**: Implements **Latent Dirichlet Allocation (LDA)** for topic extraction.
- **Clustering**: Uses **K-Means clustering** to group research papers by topics.
- **Visualization**: Generates **word clouds, topic distributions, and clustering results**.
- **Evaluation**: Uses **coherence score and silhouette score** to measure model effectiveness.

## 📂 Dataset
- **arXiv Scientific Research Papers Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset)
- Data Includes: **Titles, abstracts, authors, categories, submission dates**

## 🛠 Tech Stack
### Programming Language & Frameworks
- **Python** (Primary language)
- **Jupyter Notebook** (For experimentation and visualization)

### Libraries & Tools
- **NLTK** & **spaCy**: For text preprocessing
- **Scikit-learn**: TF-IDF, K-Means, evaluation metrics
- **Gensim**: LDA topic modeling
- **BERTopic** (Optional): For advanced NLP-based topic extraction
- **Matplotlib & Seaborn**: Data visualization
- **WordCloud**: Generates topic-based word clouds

## 🚀 Installation & Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/mnathuw/Topic-Modeling-Research-Papers.git
   cd Topic-Modeling-Research-Papers
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the notebook:**
   ```sh
   jupyter notebook
   ```
   Open the notebook and execute the cells to preprocess data, train models, and visualize results.

## 📊 Evaluation Metrics
- **Coherence Score**: Measures the interpretability of topics.
- **Silhouette Score**: Evaluates clustering quality.
- **Perplexity**: Estimates how well LDA predicts new data.

## 🔍 Future Work
- Implement **Deep Learning** methods (BERT, GPT) for contextual topic modeling.
- Apply **Graph-based techniques** for author-topic relationships.
- Extend analysis to **multi-lingual research papers**.

## 🤝 Collaborators
[@Mat3jP](https://github.com/Mat3jP) 
[@Anahita0712](https://github.com/Anahita0712) 
[@mnathuw](https://github.com/mnathuw) 
[@ImSomniac](https://github.com/ImSomniac) 
[@rafaehashmi]
(https://github.com/rafaehashmi)
@Member6

## 📜 License
This project is open-source under the **MIT License**.

---

📧 **For inquiries or contributions, feel free to reach out!** 🚀
