# 🎬 Content-Based Movie Recommender System

This project builds a **Content-Based Movie Recommendation System** using data from the **TMDB 5000 Movies and Credits** datasets.  
It recommends movies that are *most similar* to a given movie based on features like **genres**, **cast**, **crew**, and **keywords** using **NLP vectorization** and **cosine similarity**.

---

## 🚀 Project Overview

The system identifies movie similarity using a text-based representation of each movie (called `tags`), which combines:
- Movie **genres**
- Main **cast**
- **Crew** (notably the director)
- **Keywords**

These tags are vectorized using **CountVectorizer**, and recommendations are made using **cosine similarity** between movie vectors.

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computations  
- **Scikit-learn** – NLP vectorization (`CountVectorizer`) & similarity (`cosine_similarity`)  
- **Jupyter Notebook / Anaconda** – Development environment  

---

## 📂 Dataset

This project uses the **TMDB 5000 Movies Dataset** from Kaggle:
- [`tmdb_5000_movies.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [`tmdb_5000_credits.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Both files should be placed in the same directory as the notebook.

---

## ⚙️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```

2. **Install dependencies**  
   If you’re using conda:
   ```bash
   conda install pandas numpy scikit-learn
   ```
   or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

---

## 🧩 Key Steps in the Notebook

1. **Data Loading & Cleaning**
   ```python
   movies = pd.read_csv('tmdb_5000_movies.csv')
   credits = pd.read_csv('tmdb_5000_credits.csv')
   movies = movies.merge(credits, on='title')
   ```

2. **Feature Engineering**
   - Extracts genres, cast, crew, and keywords  
   - Combines them into a single `tags` column  
   - Converts tags to lowercase for normalization  

3. **Vectorization**
   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   cv = CountVectorizer(max_features=5000, stop_words='english')
   vectors = cv.fit_transform(new_df['tags']).toarray()
   ```

4. **Similarity Computation**
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   similarity = cosine_similarity(vectors)
   ```

5. **Recommendation Function**
   ```python
   def recommend(movie):
       index = new_df[new_df['title'] == movie].index[0]
       distances = similarity[index]
       movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
       for i in movie_list:
           print(new_df.iloc[i[0]].title)
   ```

---

## 🎥 Example Output

```
>>> recommend('Avatar')
John Carter
Guardians of the Galaxy
Titan A.E.
Star Trek Beyond
Aliens
```

---

## 📈 Future Improvements

- Switch from **CountVectorizer** to **TF-IDF** for better feature weighting  
- Implement a **hybrid recommender** (content + collaborative)  
- Deploy using **Streamlit** or **Flask**  
- Add movie posters using TMDB API  

---

## 🧾 Requirements

```
pandas
numpy
scikit-learn
```

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Naman Ashish**  
📧 your.email@example.com  
💼 [LinkedIn Profile or GitHub Profile]
