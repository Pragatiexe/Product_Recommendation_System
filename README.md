# Product_Recommendation_System

# 🛍️ Product Recommendation System

A hybrid product recommendation system built with AI techniques to suggest personalized product choices using content-based filtering, collaborative filtering, and an integrated chatbot assistant. This application includes a user-friendly GUI using `Tkinter` and `ttkbootstrap`.

---

## 📌 Problem Statement

With the rapid expansion of online shopping platforms, users are often overwhelmed by the large variety of available products. Customers face difficulty discovering products that best fit their preferences, leading to decision fatigue and suboptimal purchases.

---

## 💡 Proposed Solution

To address this, we present a hybrid recommendation system that analyzes product features and user behavior to suggest relevant products. The system includes a graphical interface, rating system, favorites, cart, and chatbot—all designed to enhance the shopping experience.

---

## ⚙️ System Development Approach

### 💻 Technologies Used
- **Python**: Core development
- **Tkinter + ttkbootstrap**: GUI Design
- **Pandas**: Data Handling
- **Scikit-learn**: Recommendation Algorithms
- **Matplotlib**: Data Visualization
- **OpenAI API (fallback to offline)**: Chatbot Assistant
- **SQLite (optional)**: Data logging and rating history

### 📂 Datasets
- `products.csv`: Product details (ID, name, category, brand, features, price)
- `ratings.csv`: User ratings (user_id, product_id, rating)

---

## 🧠 Algorithms Used

- **Content-Based Filtering**:  
  Recommends similar products using TF-IDF vectorization and cosine similarity based on product name, category, brand, and features.

- **User-Based Collaborative Filtering**:  
  Suggests products based on the preferences of similar users using a user-product rating matrix and cosine similarity.

- **Rule-based Offline Chatbot** *(fallback)*:  
  Provides product-related assistance through a predefined ruleset when the OpenAI API is unavailable.


## ✅ Features

- 📄 **Content-Based** and 👥 **User-Based** recommendations  
- ⭐ **Product Rating** and 🕓 **History Logging**  
- 🏆 **Top Rated Products Visualization**  
- 🔍 **Search with Filters**  
- 🛒 **Cart** and ❤️ **Favorites** Management  
- 💬 **AI Chatbot** with offline fallback  
- 🌗 **Dark/Light Theme Switcher**  
- 💾 **Persistent Logs** for recommendation tracking

---

## 🌍 Scope for Improvement

- Add **login/signup** for multi-user tracking  
- Integrate with **live product databases or APIs**  
- Include **real-time collaborative filtering with matrix factorization**  
- Deploy as a **web application (Flask/Django)**  
- Implement **email notifications or price alerts**

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [OpenAI API](https://platform.openai.com/docs)  
- [Tkinter + ttkbootstrap](https://ttkbootstrap.readthedocs.io/en/latest/)  
- [Pandas](https://pandas.pydata.org/docs/)  
- [Matplotlib](https://matplotlib.org/stable/index.html)

---

## 📦 How to Run

1. Clone this repository or download the files.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib ttkbootstrap openai
