import streamlit as st
from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError
from main import *

st.set_option('deprecation.showfileUploaderEncoding', False)
db = db_create('D:/Users/olegs/PycharmProjects/cv_l10/images')


def main():
    st.header("Computer Vision Task 10")
    st.write("Choose any image and get similar images from the dataset:")
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is None:
        pass
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = ImageOps.exif_transpose(image)
        st.image(image)
        img_opencv = np.array(image)
        img_opencv = img_opencv[:, :, ::-1].copy()
        links = get_neighbours_links(db, get_k_neighbours(vectorize(img_opencv), db, 3))
        st.success("There are similar images:")
        col = st.columns(3)
        for i in range(len(links)):
            try:
                with col[i]:
                    similar_image = Image.open('images/' + links[i])
                    st.image(similar_image, width=200)
            except UnidentifiedImageError:
                pass


if __name__ == '__main__':
    main()
