import pandas as pd
import tkinter as tk
from tkinter import messagebox, LEFT
import ttkbootstrap as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os

# Load data safely
try:
    products = pd.read_csv("products.csv")
    ratings = pd.read_csv("ratings.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# TF-IDF content similarity
products['combined_features'] = (
    products['product_name'].fillna('') + ' ' +
    products['category'].fillna('') + ' ' +
    products['brand'].fillna('') + ' ' +
    products['features'].fillna('')
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['combined_features'])
sim_df = pd.DataFrame(cosine_similarity(tfidf_matrix), index=products['product_id'], columns=products['product_id'])

# User-based matrix
user_matrix = ratings.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
user_sim = pd.DataFrame(cosine_similarity(user_matrix), index=user_matrix.index, columns=user_matrix.index)

# Logger
def log_recommend(entry_type, inp, recs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("recommendation_log.txt", "a") as f:
        f.write(f"[{ts}] {entry_type}: {inp} -> {', '.join(recs)}\n")

# Recommenders
def recommend_products(pid, num=3):
    if pid not in sim_df.index:
        return ["Product not found"]
    sims = sim_df[pid].sort_values(ascending=False).drop(pid)
    top = sims.head(num).index
    return products.loc[products['product_id'].isin(top), 'product_name'].tolist()

def recommend_for_user(uid, num=3):
    if uid not in user_sim.index:
        return ["User not found"]
    peers = user_sim[uid].sort_values(ascending=False).iloc[1:4].index
    unrated = user_matrix.loc[uid][user_matrix.loc[uid] == 0].index
    scores = {}
    for pid in unrated:
        vals = [user_matrix.loc[p, pid] for p in peers if user_matrix.loc[p, pid] > 0]
        if vals:
            scores[pid] = sum(vals) / len(vals)
    top = sorted(scores, key=scores.get, reverse=True)[:num]
    return products.loc[products['product_id'].isin(top), 'product_name'].tolist()

# Local ChatBot (Offline Mode)
def chatbot_response(user_input):
    user_input = user_input.lower()
    if any(word in user_input for word in ["hello", "hi"]):
        return "Hello! How can I help you today?"
    elif "recommend" in user_input:
        return "Try the Content or User-Based tabs to get recommendations."
    elif "cart" in user_input:
        return f"You have {len(cart)} items in your cart."
    elif "favorite" in user_input:
        return f"You have {len(favs)} items in your favorites."
    elif "thank" in user_input:
        return "You're welcome! 😊"
    else:
        return "Sorry, I didn't understand. Try asking about recommendations, cart, or favorites."

# Shared state
cart, favs = [], []

# GUI setup
root = tb.Window(themename="litera")
root.title("🛍️ Hybrid Product Recommender")
root.geometry("1000x750")

def toggle_theme():
    theme = "darkly" if root.style.theme.name == "litera" else "litera"
    root.style.theme_use(theme)
tb.Button(root, text="Toggle Theme 🌗", command=toggle_theme).pack(pady=5)

# Tabs
notebook = tb.Notebook(root)
notebook.pack(expand=True, fill='both', pady=10)
tab_names = {
    "content": "📄 Content-Based",
    "user": "👥 User-Based",
    "rate": "⭐ Rate Product",
    "history": "🕓 View History",
    "top": "🏆 Top Rated",
    "search": "🔍 Search",
    "cart": "🛒 Cart",
    "favorites": "❤️ Favorites",
    "chat": "💬 ChatBot"
}
tabs = {k: tb.Frame(notebook) for k in tab_names}
[notebook.add(frame, text=tab_names[k]) for k, frame in tabs.items()]

# Utility functions
def update_cart():
    tabs['cart'].cart_box.delete(0, 'end')
    for item in cart:
        tabs['cart'].cart_box.insert('end', item)

def update_favs():
    tabs['favorites'].fav_box.delete(0, 'end')
    for item in favs:
        tabs['favorites'].fav_box.insert('end', item)

def add_cart(item):
    if item:
        cart.append(item)
        update_cart()
        messagebox.showinfo("Cart", f"Added → {item}")

def add_fav(item):
    if item and item not in favs:
        favs.append(item)
        update_favs()
        messagebox.showinfo("Favorites", f"Liked → {item}")

# Content-Based Tab
t1 = tabs['content']
cb_var = tb.StringVar()
tb.Combobox(t1, textvariable=cb_var, values=products['product_name'].tolist(), width=50).pack(pady=10)
t1.res_box = tk.Listbox(t1, height=6, width=70)
t1.res_box.pack()
tb.Button(t1, text="Recommend", command=lambda: (
    log_recommend("Content", cb_var.get(), (recs := recommend_products(products.loc[products['product_name'] == cb_var.get(), 'product_id'].values[0]))),
    t1.res_box.delete(0, 'end'),
    [t1.res_box.insert('end', r) for r in recs]
)).pack(pady=5)
tb.Button(t1, text="Add to Cart", command=lambda: add_cart(t1.res_box.get('active'))).pack()
tb.Button(t1, text="Add to Favorites", command=lambda: add_fav(t1.res_box.get('active'))).pack()

# User-Based Tab
t2 = tabs['user']
ub_var = tb.StringVar()
tb.Combobox(t2, textvariable=ub_var, values=user_matrix.index.tolist(), width=20).pack(pady=10)
t2.res_box = tk.Listbox(t2, height=6, width=70)
t2.res_box.pack()
tb.Button(t2, text="Recommend", command=lambda: (
    log_recommend("User", ub_var.get(), (recs := recommend_for_user(int(ub_var.get())))),
    t2.res_box.delete(0, 'end'),
    [t2.res_box.insert('end', r) for r in recs]
)).pack(pady=5)
tb.Button(t2, text="Add to Cart", command=lambda: add_cart(t2.res_box.get('active'))).pack()
tb.Button(t2, text="Add to Favorites", command=lambda: add_fav(t2.res_box.get('active'))).pack()

# Rate Product Tab
t3 = tabs['rate']
tb.Label(t3, text="Select Product:").pack(pady=5)
rate_var = tb.StringVar()
tb.Combobox(t3, textvariable=rate_var, values=products['product_name'].tolist(), width=40).pack(pady=5)
rating = tk.IntVar()
stars, frame = [], tk.Frame(t3)
frame.pack()

def select_star(n):
    rating.set(n)
    for i in range(5):
        stars[i].config(text="★" if i < n else "☆")

for i in range(5):
    b = tk.Button(frame, text="☆", command=lambda i=i: select_star(i+1), font=("Arial", 18))
    b.pack(side='left', padx=2)
    stars.append(b)

def save_rating():
    pn = rate_var.get()
    r = rating.get()
    if pn and r > 0:
        pid = products.loc[products['product_name'] == pn, 'product_id'].iloc[0]
        pd.DataFrame([[1, pid, r]], columns=['user_id', 'product_id', 'rating']).to_csv("ratings.csv", mode='a', index=False, header=False)
        messagebox.showinfo("Saved", f"Rated {pn} → {r}★")
    else:
        messagebox.showerror("Oops", "Select product & rating first!")

tb.Button(t3, text="Submit", command=save_rating).pack(pady=10)

# History Tab
t4 = tabs['history']
t4.txt = tk.Text(t4, width=95, height=30)
t4.txt.pack(pady=5)
if os.path.exists("recommendation_log.txt"):
    t4.txt.insert('end', open("recommendation_log.txt").read())

# Top Rated Tab
t5 = tabs['top']
top_grp = ratings.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(5)
names = products.loc[products['product_id'].isin(top_grp.index), 'product_name'].tolist()
vals = top_grp.values.tolist()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(names, vals, color='skyblue')
ax.set_ylabel("Avg Rating")
ax.set_title("Top 5 Products")
ax.tick_params(axis='x', rotation=45)
canvas = FigureCanvasTkAgg(fig, master=t5)
canvas.draw()
canvas.get_tk_widget().pack(pady=10)

# Search Tab
t6 = tabs['search']
tk.Label(t6, text="Search:").pack(pady=5)
search_var = tb.StringVar()
tb.Entry(t6, textvariable=search_var, width=60).pack()
result_box = tk.Listbox(t6, width=80, height=20)
result_box.pack(pady=10)

def do_search():
    kw = search_var.get().lower()
    df = products[products['product_name'].str.lower().str.contains(kw)]
    result_box.delete(0, 'end')
    for _, r in df.iterrows():
        result_box.insert('end', f"{r['product_name']} - ₹{r.get('price', 0):.2f}")

tb.Button(t6, text="Go", command=do_search).pack()

# Cart Tab
t7 = tabs['cart']
t7.cart_box = tk.Listbox(t7, width=60, height=20)
t7.cart_box.pack(pady=10)
tb.Button(t7, text="Clear Cart", command=lambda: (cart.clear(), update_cart())).pack()

# Favorites Tab
t8 = tabs['favorites']
t8.fav_box = tk.Listbox(t8, width=60, height=20)
t8.fav_box.pack(pady=10)
tb.Button(t8, text="Clear Favorites", command=lambda: (favs.clear(), update_favs())).pack()

# ChatBot Tab (Offline)
t9 = tabs['chat']
t9.chat = tk.Text(t9, width=80, height=25, wrap='word')
t9.chat.pack(pady=10)
t9.chat.config(state='disabled')
chat_var = tb.StringVar()

def chat_send():
    q = chat_var.get().strip()
    if q:
        t9.chat.config(state='normal')
        t9.chat.insert('end', f"You: {q}\n")
        response = chatbot_response(q)
        t9.chat.insert('end', f"Bot: {response}\n\n")
        t9.chat.config(state='disabled')
        t9.chat.see('end')
        chat_var.set("")

entry_frame = tb.Frame(t9)
entry_frame.pack(pady=5)
tb.Entry(entry_frame, textvariable=chat_var, width=60).pack(side=LEFT, padx=5)
tb.Button(entry_frame, text="Send", command=chat_send, bootstyle="primary").pack(side=LEFT, padx=5)

# Run app
root.mainloop()
