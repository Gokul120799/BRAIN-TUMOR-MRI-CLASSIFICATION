
import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/content"

CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "resnet_model.h5")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.pkl")


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image and classify the tumor type "
    "using the trained deep learning models."
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Custom CNN Model": CNN_MODEL_PATH,
    "ResNet50 Model": RESNET_MODEL_PATH,
    "Best Model": BEST_MODEL_PATH,
    "Class Indices": CLASS_INDICES_PATH
}

missing_files = []

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):
        missing_files.append(
            f"{file_name}: {file_path}"
        )

if missing_files:

    st.error("❌ Some required files are missing:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    try:

        cnn = load_model(
            CNN_MODEL_PATH,
            compile=False
        )

        resnet = load_model(
            RESNET_MODEL_PATH,
            compile=False
        )

        best = load_model(
            BEST_MODEL_PATH,
            compile=False
        )

        return cnn, resnet, best

    except Exception as e:

        st.error(
            f"❌ Error loading models:\n\n{str(e)}"
        )

        return None, None, None


cnn_model, resnet_model, best_model = load_models()


# ============================================================
# LOAD CLASS INDICES
# ============================================================

@st.cache_data
def load_class_labels():

    try:

        with open(
            CLASS_INDICES_PATH,
            "rb"
        ) as f:

            class_indices = pickle.load(f)

        # Example:
        # {'glioma': 0, 'meningioma': 1, ...}

        class_labels = {
            value: key
            for key, value in class_indices.items()
        }

        return class_labels

    except Exception as e:

        st.error(
            f"❌ Error loading class_indices.pkl:\n\n{str(e)}"
        )

        return {}


class_labels = load_class_labels()


# ============================================================
# SHOW MODEL INFORMATION
# ============================================================

with st.expander("📁 Model Files"):

    st.write(
        f"**CNN Model:** `{CNN_MODEL_PATH}`"
    )

    st.write(
        f"**ResNet Model:** `{RESNET_MODEL_PATH}`"
    )

    st.write(
        f"**Best Model:** `{BEST_MODEL_PATH}`"
    )

    st.write(
        f"**Class Indices:** `{CLASS_INDICES_PATH}`"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img_file):

    img = image.load_img(
        img_file,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    return img_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_tumor(model, img_array):

    if model is None:

        return "Model Not Available", 0.0

    try:

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predictions = np.asarray(predictions)

        # ----------------------------------------------------
        # MULTICLASS MODEL
        # ----------------------------------------------------

        if predictions.ndim == 2 and predictions.shape[1] > 1:

            pred_class = int(
                np.argmax(
                    predictions[0]
                )
            )

            confidence = float(
                np.max(
                    predictions[0]
                )
            )

        # ----------------------------------------------------
        # BINARY MODEL
        # ----------------------------------------------------

        else:

            probability = float(
                predictions.flatten()[0]
            )

            if probability >= 0.5:

                pred_class = 1
                confidence = probability

            else:

                pred_class = 0
                confidence = 1.0 - probability

        tumor_name = class_labels.get(
            pred_class,
            f"Class {pred_class}"
        )

        return tumor_name, confidence

    except Exception as e:

        st.error(
            f"❌ Prediction error:\n\n{str(e)}"
        )

        return "Prediction Error", 0.0


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DISPLAY UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    st.subheader("🖼️ Uploaded MRI Image")

    st.image(
        uploaded_file,
        caption="Uploaded Brain MRI",
        width="stretch"
    )


# ============================================================
# MODEL SELECTION
# ============================================================

model_choice = st.selectbox(
    "🤖 Choose Prediction Model",
    [
        "Custom CNN",
        "ResNet50 Transfer Learning",
        "Best Model"
    ]
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔍 Predict Tumor Type",
    type="primary"
):

    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload a brain MRI image first."
        )

    else:

        with st.spinner(
            "Analyzing MRI image..."
        ):

            img_array = preprocess_image(
                uploaded_file
            )

            # ------------------------------------------------
            # SELECT MODEL
            # ------------------------------------------------

            if model_choice == "Custom CNN":

                tumor_type, confidence = predict_tumor(
                    cnn_model,
                    img_array
                )

            elif model_choice == "ResNet50 Transfer Learning":

                tumor_type, confidence = predict_tumor(
                    resnet_model,
                    img_array
                )

            else:

                tumor_type, confidence = predict_tumor(
                    best_model,
                    img_array
                )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            f"🧠 Predicted Tumor Type: **{tumor_type}**"
        )

        st.info(
            f"📊 Confidence Score: **{confidence * 100:.2f}%**"
        )

        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Brain Tumor MRI Classification using Deep Learning"
)
