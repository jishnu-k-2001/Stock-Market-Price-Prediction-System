import streamlit as st

# set page config
st.set_page_config(page_title="Stock Price Analyser", page_icon=":moneybag:", layout="wide")

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


# add sidebar
st.sidebar.title("Language Selection")
Language = st.sidebar.selectbox('Select Language', ['English', 'Malayalam'], key='lang')

# add content
st.title("Stock Learning Basics")

st.header("Start to learn stock market")
st.write("Step-1")
if Language == "English" :
    st.video("https://www.youtube.com/watch?v=GpiM_qi5mAc")
else:
    st.video("https://www.youtube.com/watch?v=QlUS5mUSHjo")

st.title("Cryptocurrency Basics")

st.header("Start to learn Cryptocurrency")
st.write("Step-1")
if Language == "English" :
    st.video("https://www.youtube.com/watch?v=1YyAzVmP9xQ")
else:
    st.video("https://www.youtube.com/watch?v=Xv4xgoGjAC8")