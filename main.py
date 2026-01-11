import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
# ----------------------------------------------------------
# 🎯 Hybrid Recommendation Function
# ----------------------------------------------------------
def hybrid_recommend(
    target_user_id,
    user_knn,
    book_knn,
    user_book_matrix,
    book_user_matrix,
    user_index_mapping,
    user_id_mapping,
    book_index_mapping,
    book_mapping,
    top_n=10
):
    if target_user_id not in user_index_mapping:
        st.warning("❌ User not found in mapping.")
        return []

    # Step A: Find similar users
    target_user_index = user_index_mapping[target_user_id]
    distances, indices = user_knn.kneighbors(user_book_matrix[target_user_index], n_neighbors=6)
    similar_users = [user_id_mapping[i] for i in indices.flatten() if i != target_user_index]

    # Step B: Books read by target user
    target_user_books = user_book_matrix[target_user_index].nonzero()[1]

    # Step C: Books liked by similar users
    similar_users_books = set()
    for user in similar_users:
        sim_user_index = user_index_mapping[user]
        row = user_book_matrix[sim_user_index]
        liked_books = row.indices[row.data > 4]
        for b in liked_books:
            if b not in target_user_books:
                similar_users_books.add(b)

    # Step D: Books similar to what user has read
    similar_books_set = set()
    for b in target_user_books:
        distances, indices = book_knn.kneighbors(book_user_matrix[b], n_neighbors=4)
        similar_books_set.update(indices.flatten()[1:])

    # Step E: Combine both sources
    final_recommendations = list(similar_users_books.union(similar_books_set))[:top_n]

    # Step F: Compute average rating for each recommended book
    recs = []
    for i in final_recommendations:
        ratings = book_user_matrix[i].data
        avg_rating = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
        recs.append((book_mapping[i], round(avg_rating, 2)))

    return recs


# ----------------------------------------------------------
# ⚙️ Load Pre-trained Models and Data
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    base_path = "models/"
    user_knn = joblib.load(base_path + "user_knn.joblib")
    book_knn = joblib.load(base_path + "book_knn.joblib")
    user_book_matrix = joblib.load(base_path + "user_book_matrix.joblib")
    book_user_matrix = joblib.load(base_path + "book_user_matrix.joblib")
    user_index_mapping = joblib.load(base_path + "user_index_mapping.joblib")
    user_id_mapping = joblib.load(base_path + "user_id_mapping.joblib")
    book_index_mapping = joblib.load(base_path + "book_index_mapping.joblib")
    book_mapping = joblib.load(base_path + "book_mapping.joblib")

    # Load your books dataset that contains image URLs
    books_df = pd.read_csv("data/Books.csv", usecols=["Book-Title", "Image-URL-L"],encoding='latin1')

    return (
        user_knn,
        book_knn,
        user_book_matrix,
        book_user_matrix,
        user_index_mapping,
        user_id_mapping,
        book_index_mapping,
        book_mapping,
        books_df
    )


# ----------------------------------------------------------
# 🖼️ Display Book Covers in a Modern Grid
# ----------------------------------------------------------

def display_book_grid(recommendations, books_df):
    if not recommendations:
        st.warning("No recommendations found.")
        return

    st.subheader("📚 Recommended Books For You")

    num_cols = 4  # 4 books per row
    fallback_path = "images/no_image.png"

    # Ensure fallback image exists
    if not os.path.exists(fallback_path):
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        img = Image.new("RGB", (150, 220), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = "No Image"

        try:
            if font is not None:
                bbox = draw.textbbox((0, 0), text, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                raise AttributeError
        except Exception:
            w, h = (len(text) * 6, 11)

        x, y = (150 - w) / 2, (220 - h) / 2
        try:
            draw.text((x, y), text, fill=(90, 90, 90), font=font)
        except Exception:
            draw.text((x, y), text, fill=(90, 90, 90))
        img.save(fallback_path)

    # Helper: Check if image is mostly white
    def is_blank_image(image, threshold=245, white_ratio=0.98):
        """Detects if the image is mostly white or blank."""
        try:
            img_gray = image.convert("L")
            arr = np.array(img_gray)
            white_pixels = np.mean(arr > threshold)
            return white_pixels > white_ratio
        except Exception:
            return False

    # Function to safely load image from a URL or fallback
    def load_image_or_fallback(url):
        try:
            if url and isinstance(url, str) and url.startswith("http"):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/141.0.0.0 Safari/537.36"
                    )
                }
                response = requests.get(url, headers=headers, timeout=10)
                if (
                    response.status_code == 200
                    and "image" in response.headers.get("Content-Type", "")
                ):
                    img = Image.open(BytesIO(response.content))
                    if is_blank_image(img):
                        return Image.open(fallback_path)
                    return img
            return Image.open(fallback_path)
        except Exception:
            try:
                return Image.open(fallback_path)
            except Exception:
                return Image.new("RGB", (150, 220), color=(240, 240, 240))

    # Display books in 4xN grid neatly
    for i in range(0, len(recommendations), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j >= len(recommendations):
                break

            book_title, _ = recommendations[i + j]
            row = books_df[books_df["Book-Title"] == book_title]
            img_url = row["Image-URL-L"].values[0] if not row.empty else None

            img = load_image_or_fallback(img_url)
            try:
                img = img.convert("RGB")
            except Exception:
                pass

            with col:
                st.image(img, use_container_width=True)
                st.markdown(
                    f"""
                    <div style="text-align:center; height:40px; 
                                display:flex; align-items:center; 
                                justify-content:center; margin-top:8px;">
                        <strong style="font-size:14px; line-height:1.2;">
                            {book_title}
                        </strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



# ----------------------------------------------------------
# 🚀 Streamlit UI
# ----------------------------------------------------------
st.title("📖 Book Recommendation System")

with st.spinner("Loading pre-trained models..."):
    (
        user_knn,
        book_knn,
        user_book_matrix,
        book_user_matrix,
        user_index_mapping,
        user_id_mapping,
        book_index_mapping,
        book_mapping,
        books_df,
    ) = load_models()
st.success("✅ Models and data loaded successfully!")


# ----------------------------------------------------------
# 🔍 User Input
# ----------------------------------------------------------
user_input = st.text_input("Enter your User ID (1 - 278858):")

if user_input.strip():
    try:
        user_id = int(user_input.strip())
        if user_id in user_index_mapping:
            st.success(f"User ID {user_id} found ✅")

            if st.button("Get Recommendations"):
                with st.spinner("Generating hybrid recommendations..."):
                    recommendations = hybrid_recommend(
                        user_id,
                        user_knn,
                        book_knn,
                        user_book_matrix,
                        book_user_matrix,
                        user_index_mapping,
                        user_id_mapping,
                        book_index_mapping,
                        book_mapping,
                        top_n=12,
                    )
                    display_book_grid(recommendations, books_df)
        else:
            st.error("User ID not found. Please try again.")
    except ValueError:
        st.error("Please enter a valid integer User ID.")
else:
    st.info("Please enter your User ID to begin.")

st.sidebar.info("Developed by **Siddhesh, Dhanush, Samadhan, Prathmesh, Gayathri, Esha** 🧠")
