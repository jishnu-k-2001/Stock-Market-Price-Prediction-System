
import pymongo
import streamlit as st
from PIL import Image
from io import BytesIO

navbar_style = """
    <style>
        .navbar {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            background-color: #FF0000;
            padding: 10px;
            margin-bottom: 30px;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            border-radius: 5px;
        }
    </style>
"""

# Render navbar
st.markdown(navbar_style, unsafe_allow_html=True)
st.container().markdown('<div class="navbar">Stock Price Analyzer</div>', unsafe_allow_html=True)

# Connect to MongoDB
def connect_to_db():
    username = 'stockpriceanalyser'
    password = 'stockpriceanalyser'
    dbname = 'stockpriceanalyser'
    clustername = 'Cluster0'
    client = pymongo.MongoClient(f"mongodb+srv://stockpriceanalyser:stockpriceanalyser@cluster0.2ch0w8v.mongodb.net/?retryWrites=true&w=majority")
    return client[dbname]

db = connect_to_db()

name = st.text_input('Enter your name:')
review = st.text_area('Enter your review:')
image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

reviews_displayed = False

if st.button('Submit'):
    if name == '' or review == '':
        st.error('Enter the data')
    else:
        # Convert the uploaded image to binary data
        image_data = image_file.read()
        image = Image.open(BytesIO(image_data))
        binary_image = BytesIO()
        image.save(binary_image, format='PNG')
        db.reviews.insert_one({'name': name, 'review': review, 'image': binary_image.getvalue()})
        st.success('Review submitted!')

if st.button('View Past Reviews'):
    reviews_displayed = True
    reviews = db.reviews.find()
    for review in reviews:
        st.write(review['name'] + ': ' + review['review'])
        if 'image' in review:
            st.image(Image.open(BytesIO(review['image'])), caption=review['name'], use_column_width=True)

if st.button("Clear Reviews"):
    if reviews_displayed:
        st.session_state.reviews_cleared = True
    else:
        st.error("No reviews to clear!")
        
if st.session_state.get("reviews_cleared"):
    st.session_state.pop("reviews_cleared")
    st.success("Past reviews cleared!")

 