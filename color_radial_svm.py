# © 2024 Numantic Solutions
# https://github.com/Numantic-NMoroney
# MIT License
#

import numpy as np
import colour
import matplotlib.pyplot as plt
import streamlit as st
import zipfile

from sklearn import svm

# print("streamlit : color radial svm - 02")


def lightness_slice(min_, max_, n, lightness):
    x = np.linspace(min_, max_, n)
    y = np.linspace(min_, max_, n)

    aa, bb = np.meshgrid(x, y)

    ll = x = np.empty(aa.shape)
    ll.fill(lightness)

    return list(zip(ll.flatten(), aa.flatten(), bb.flatten()))

def a_star_slice(min_, max_, b_n, l_n, a_star):
    x = np.linspace(min_, max_, b_n)
    y = np.linspace(0, 100, l_n)

    bb, ll = np.meshgrid(x, y)

    aa = np.empty(bb.shape)
    aa.fill(a_star)

    return list(zip(ll.flatten(), aa.flatten(), bb.flatten()))

def in_srgb_gamut(labs):
    labs_in, srgbs_in = [], []
    for lab in labs:
        xyz = colour.Lab_to_XYZ(lab)
        srgb = colour.XYZ_to_sRGB(xyz)
        clipped = np.clip(srgb, 0, 1)
        if np.array_equal(srgb, clipped):
            labs_in.append(lab)
            srgbs_in.append(srgb)
    return np.array(labs_in), np.array(srgbs_in)


if 'astar' not in st.session_state:
    st.session_state.astar = 0
if 'steps' not in st.session_state:
    st.session_state.steps = 80
if 'show' not in st.session_state:
    st.session_state.show = False

@st.cache_data
def prep_model() : 

    # print("  prep model")
    path_zip = "data/"
    name_zip = "ml_color-11_terms-min_670-rgbn.tsv.zip"
    lines = []
    with zipfile.ZipFile(path_zip + name_zip) as archive :
      item = archive.read(name_zip[:-4])
      s = item.decode()
      lines = s.split('\n')


    rs, gs, bs, names = [], [], [], []
    for line in lines :
      ts = line.split('\t')
      if len(ts) == 4 :
        rs.append(int(ts[0]))
        gs.append(int(ts[1]))
        bs.append(int(ts[2]))
        names.append(ts[3])

    unique_names = list(set(names))

    to_index, to_name = {}, {}
    i = 0
    for name in unique_names :
      to_index[name] = i
      to_name[i] = name
      i += 1

    classes = []
    for name in names :
      classes.append(to_index[name])

    rgbs = list(zip(rs, gs, bs))

    f1 = open("data/ml_color-11_terms-centroids_rgbn.tsv", "r")
    to_centroids = {}
    for line in f1:
        ts = line.strip().split()
        if len(ts) > 0:
            rgb = [ int(ts[0]), int(ts[1]), int(ts[2]) ]
            to_centroids[ts[3]] = rgb
    f1.close()

    classifier = svm.SVC(kernel='rbf')
    classifier.fit(rgbs, classes)

    return classifier, to_name, to_centroids


clf, to_name, to_centroids = prep_model()

slice_ = a_star_slice(-120, 120, 
                      st.session_state.steps, 
                      int(st.session_state.steps/2),
                      st.session_state.astar)

labs, srgbs = in_srgb_gamut(slice_)


if not st.session_state.show:
    for i in range(srgbs.shape[0]):
        rgb_in = ( srgbs[i,0] * 255, srgbs[i,1] * 255, srgbs[i,2] * 255 )

        prediction = clf.predict( [rgb_in] )

        prediction_name = to_name[prediction.item()]
        centroid = to_centroids[prediction_name]
        srgbs[i,0] = float(centroid[0]) / 255.0
        srgbs[i,1] = float(centroid[1]) / 255.0
        srgbs[i,2] = float(centroid[2]) / 255.0
    

st.subheader("Color Radial SVM")
st.markdown("A radial [support vector machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) color classifier.")
st.markdown("Input data is a constant a* [CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space) slice (sampling inside the [sRGB](https://en.wikipedia.org/wiki/SRGB) [gamut](https://en.wikipedia.org/wiki/Gamut)).")
st.markdown("Labels are : *red, green, yellow, blue, purple, pink, orange, brown, black, gray & white*.")

# col1, col2 = st.columns([0.99, 0.01])
col1, col2 = st.columns([0.8, 0.2])

with col1:

    col_a, col_b, col_c = st.columns([0.33, 0.33, 0.33])
    with col_a:
        _ = st.slider("a* : ", -83, 93, 0, key='astar')
    with col_b:
        _ = st.slider("Steps : ", 11, 101, 80, key='steps')
    with col_c:
        on = st.toggle("Show input colors", key='show')

    plt.scatter(labs[:,2], labs[:,0], c=srgbs)
    plt.xlabel('b*')
    plt.ylabel('L*')
    plt.title('CIELAB a* Slice')
    plt.axis('equal')

    with st.spinner(""):
        st.pyplot(plt.gcf())

        st.markdown("[**CIC 32**](https://www.imaging.org/IST/IST/Conferences/CIC/CIC2024/CIC_Home.aspx) &mdash; [**Courses & Workshops**](https://www.imaging.org/IST/Conferences/CIC/CIC2024/CIC_Home.aspx?WebsiteKey=6d978a6f-475d-46cc-bcf2-7a9e3d5f8f82&8a93a38c6b0c=3#8a93a38c6b0c) &mdash; **More Apps:** [KNN](https://knearestcolors.streamlit.app/), [Color Forest](https://randomcolorforest.streamlit.app/), [Zip Highlight](https://ziphighlighter.streamlit.app) ")



