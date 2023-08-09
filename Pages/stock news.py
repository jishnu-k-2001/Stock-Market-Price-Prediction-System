import streamlit as st
from GoogleNews import GoogleNews
import pandas as pd

# Define CSS style for navbar
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

def get_news():
    googlenews = GoogleNews()
    googlenews.search('Stock market')
    result = googlenews.result()
    df = pd.DataFrame(result)
    return df

st.title('Stock Market News')
df = get_news()
for index, row in df.iterrows():
    st.subheader(row['title'])
    st.write(row['desc'])
    st.write(row['link'])
    st.write('Published Date:', row['date'])
    st.write('-----------------------------------------')