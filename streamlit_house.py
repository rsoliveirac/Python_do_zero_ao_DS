import streamlit as st
import pandas as pd
import folium
import numpy as np

from streamlit_folium import folium_static

from folium.plugins import MarkerCluster

st.set_page_config( layout = 'wide')

@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )

    return data

#get add

path = '/home/user/repos/python_do_zero_ao_DS/dataset/kc_house_data.csv'
data = get_data(path)

f_attributes = st.sidebar.multiselect( 'Enter columns',
data.columns )
f_zipcode = st.sidebar.multiselect(
 'Enter zipcode',
 data['zipcode'].unique() )

