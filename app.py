from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# BRAIN TUMOR MRI CLASSIFICATION
# Custom CNN Experiment 2 + ResNet50 Experiment 3
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Custom CNN": (
        APP_DIR
        / "fresh_training_outputs"
        / "models"
        / "fresh_custom_cnn_exp2.keras"
    ),
    "ResNet50": (
        APP_DIR
        / "experiment3_resnet50"
        / "models"
        / "best_resnet50_exp3.keras"
    ),
}

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.80

# Recorded evaluation results from this project's test set.
# These are dataset-level results, not guarantees for an uploaded image.
MODEL_RESULTS = {
    "Custom CNN": {
        "experiment": "Experiment 2",
        "accuracy": "84.96%",
        "macro_f1": "83.89%",
        "incorrect": "37 / 246",
    },
    "ResNet50": {
        "experiment": "Experiment 3",
        "accuracy": "92.68%",
        "macro_f1": "92.01%",
        "incorrect": "18 / 246",
    },
}


# ============================================================
# 1. LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading selected model...")
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}. "
            "Expected (None, 224, 224, 3)."
        )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Unexpected model output shape: {model.output_shape}. "
            f"Expected {len(CLASS_NAMES)} classes."
        )

    return model


# ============================================================
# 2. OPEN AND PREPROCESS THE UPLOADED MRI
# ============================================================

def prepare_image(uploaded_file):
    """Open the uploaded image and create model-specific input batches."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Keep the same 224 x 224 input size and nearest-neighbour
    # resizing approach used in the existing application.
    resized_image = image.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.NEAREST,
    )

    image_array = np.asarray(resized_image, dtype=np.float32)

    # Custom CNN training pipeline used rescaling to the 0-1 range.
    custom_cnn_batch = np.expand_dims(image_array / 255.0, axis=0)

    # ResNet50 training pipeline used its own preprocessing function.
    # Do NOT divide this input by 255.
    resnet_batch = np.expand_dims(
        preprocess_input(image_array.copy()),
        axis=0,
    )

    return image, resized_image, custom_cnn_batch, resnet_batch


# ============================================================
# 3. RUN AND VALIDATE A PREDICTION
# ============================================================

def predict_image(model, image_batch):
    output = np.asarray(model.predict(image_batch, verbose=0))

    if output.shape != (1, len(CLASS_NAMES)):
        raise ValueError(
            f"Expected prediction shape (1, {len(CLASS_NAMES)}), "
            f"but received {output.shape}."
        )

    scores = output[0]

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction values.")

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("The model output is outside the expected 0-1 range.")

    if not np.isclose(scores.sum(), 1.0, atol=1e-3):
        raise ValueError(
            "The model output does not sum approximately to 1. "
            "The output may not be a softmax probability vector."
        )

    predicted_index = int(np.argmax(scores))

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": float(scores[predicted_index]),
        "scores": scores,
    }


# ============================================================
# 4. DISPLAY ONE MODEL'S RESULTS
# ============================================================

def display_prediction(model_name, result):
    predicted_class = result["class"]
    confidence = result["confidence"]
    scores = result["scores"]

    st.markdown(f"### {model_name}")

    result_col, confidence_col = st.columns(2)

    result_col.metric(
        "Predicted class",
        predicted_class.replace("_", " ").title(),
    )
    confidence_col.metric(
        "Highest softmax score",
        f"{confidence * 100:.2f}%",
    )

    st.progress(
        confidence,
        text=f"Highest softmax score: {confidence * 100:.2f}%",
    )

    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(
            "This prediction meets the application's initial 80% "
            "review threshold."
        )
    else:
        st.info(
            "Review note: this score is below the application's initial "
            "80% guideline. Consider the image alongside other relevant "
            "information as part of a careful review."
        )

    score_table = {
        "Class": [
            name.replace("_", " ").title()
            for name in CLASS_NAMES
        ],
        "Softmax score (%)": [
            round(float(value) * 100, 2)
            for value in scores
        ],
    }

    st.dataframe(
        score_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Softmax scores describe the model's output distribution. "
        "They are not calibrated probabilities and do not guarantee "
        "that the predicted class is correct."
    )


# ============================================================
# 5. PAGE HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image and classify it using either of the "
    "two models trained for this project, or compare both models "
    "on the same image."
)

with st.expander("About the two trained models"):
    st.markdown(
        """
        **Custom CNN – Experiment 2**
        - Four-class custom convolutional neural network.
        - Input size: 224 × 224 × 3.
        - Preprocessing: pixel values scaled to the 0–1 range.
        - Recorded test accuracy: 84.96%.
        - Recorded test macro F1-score: 83.89%.

        **ResNet50 – Experiment 3**
        - ImageNet-pretrained ResNet50 fine-tuned for this project.
        - Input size: 224 × 224 × 3.
        - Preprocessing: `resnet50.preprocess_input`.
        - Recorded test accuracy: 92.68%.
        - Recorded test macro F1-score: 92.01%.

        ResNet50 was selected as the best-performing model based on
        the recorded test-set comparison. These metrics describe the
        project test set and do not guarantee performance on a new image.
        """
    )

# ============================================================
# 6. MODEL SELECTION
# ============================================================

st.subheader("1. Select the model")

model_choice = st.radio(
    "Which model would you like to use?",
    options=[
        "ResNet50",
        "Custom CNN",
        "Compare Both Models",
    ],
    index=0,
    horizontal=True,
    help=(
        "Choose one model for a single prediction, or compare both "
        "models using the same uploaded MRI image."
    ),
)

# ============================================================
# 7. CHECK REQUIRED MODEL FILES AND LOAD SELECTED MODELS
# ============================================================

selected_model_names = (
    ["Custom CNN", "ResNet50"]
    if model_choice == "Compare Both Models"
    else [model_choice]
)

missing_models = [
    name for name in selected_model_names
    if not MODEL_PATHS[name].is_file()
]

if missing_models:
    st.error(
        "The following required model file(s) could not be found: "
        + ", ".join(missing_models)
    )
    for name in missing_models:
        st.code(str(MODEL_PATHS[name]))

    st.info(
        "Keep app.py in the project root and check that the model folders "
        "match the expected structure shown above."
    )
    st.stop()

loaded_models = {}

for name in selected_model_names:
    try:
        loaded_models[name] = load_model(str(MODEL_PATHS[name]))
    except Exception as exc:
        st.error(f"Could not load the {name} model.")
        st.exception(exc)
        st.stop()


# ============================================================
# 8. UPLOAD MRI IMAGE
# ============================================================

st.divider()
st.subheader("2. Upload a brain MRI image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

if uploaded_file is not None:
    try:
        original_image, preview_image, custom_batch, resnet_batch = (
            prepare_image(uploaded_file)
        )

        image_col, details_col = st.columns([1, 1])

        with image_col:
            st.image(
                original_image,
                caption="Uploaded MRI image",
                use_container_width=True,
            )

        with details_col:
            st.markdown("**Image details**")
            st.write(f"Filename: {uploaded_file.name}")
            st.write(f"Original image size: {original_image.size[0]} × {original_image.size[1]}")
            st.write("Model input size: 224 × 224 pixels")
            st.write("Colour format: RGB")

        st.divider()
        st.subheader("3. Run classification")

        if st.button(
            "🔍 Predict MRI Class",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Analysing MRI image..."):
                    results = {}

                    for name, model in loaded_models.items():
                        if name == "Custom CNN":
                            model_input = custom_batch
                        else:
                            model_input = resnet_batch

                        results[name] = predict_image(model, model_input)

                st.subheader("4. Prediction results")

                if model_choice == "Compare Both Models":
                    cnn_col, resnet_col = st.columns(2)

                    with cnn_col:
                        display_prediction(
                            "Custom CNN – Experiment 2",
                            results["Custom CNN"],
                        )

                    with resnet_col:
                        display_prediction(
                            "ResNet50 – Experiment 3",
                            results["ResNet50"],
                        )

                    cnn_class = results["Custom CNN"]["class"]
                    resnet_class = results["ResNet50"]["class"]

                    st.divider()
                    st.markdown("### Model agreement")

                    if cnn_class == resnet_class:
                        st.success(
                            "Both models predicted the same class: "
                            f"**{cnn_class.replace('_', ' ').title()}**."
                        )
                    else:
                        st.info(
                            "Review note: the models produced different "
                            "predictions. Custom CNN suggested "
                            f"**{cnn_class.replace('_', ' ').title()}**, "
                            "and ResNet50 suggested "
                            f"**{resnet_class.replace('_', ' ').title()}**. "
                            "This comparison provides a useful opportunity "
                            "for closer review."
                        )

                    st.info(
                        "Project note: agreement shows that both models "
                        "identified the same class for this image. It is "
                        "a helpful comparison result; professional medical "
                        "interpretation remains essential."
                    )

                else:
                    display_prediction(
                        f"{model_choice} – {MODEL_RESULTS[model_choice]['experiment']}",
                        results[model_choice],
                    )

            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)

    except Exception as exc:
        st.error(
            "Could not open or preprocess this file. "
            "Please upload a valid JPG, JPEG or PNG image."
        )
        st.exception(exc)


# ============================================================
# 9. SIDEBAR: RECORDED MODEL COMPARISON
# ============================================================

with st.sidebar:
    st.header("Model comparison")
    st.caption("Recorded results on this project's test set")

    for name in ["Custom CNN", "ResNet50"]:
        details = MODEL_RESULTS[name]
        st.markdown(f"**{name} – {details['experiment']}**")
        st.metric("Test accuracy", details["accuracy"])
        st.write(f"Macro F1-score: {details['macro_f1']}")
        st.write(f"Incorrect predictions: {details['incorrect']}")
        st.divider()

    st.markdown("**Selected best-performing model:** ResNet50")
    st.write(f"Initial review guideline: {CONFIDENCE_THRESHOLD * 100:.0f}%")

    with st.expander("Model file locations"):
        st.write("Custom CNN")
        st.code(str(MODEL_PATHS["Custom CNN"]))
        st.write("ResNet50")
        st.code(str(MODEL_PATHS["ResNet50"]))


# ============================================================
# 10. RESPONSIBLE USE
# ============================================================

st.divider()
st.subheader("Responsible use")

st.info(
    "This application demonstrates brain MRI image classification "
    "for educational and project purposes. The model's results support "
    "learning and technical evaluation. Softmax scores are model outputs, "
    "not calibrated guarantees of correctness; the 80% guideline helps "
    "organise review and does not confirm a diagnosis. For health-related "
    "decisions, consult qualified medical professionals. Any future "
    "clinical application would require independent medical validation "
    "and appropriate professional oversight."
)

st.caption(
    "Brain Tumor MRI Classification | Custom CNN Experiment 2 "
    "+ ResNet50 Experiment 3"
)
