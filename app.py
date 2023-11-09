import os
import json
import streamlit as st
import base64
import yaml
from KidneyDiseaseClassifier.utils.common import decodeImage
from KidneyDiseaseClassifier.pipeline.prediction import PredictionPipeline
from KidneyDiseaseClassifier.config.configuration import ConfigurationManager

# Initialize the Streamlit app
st.title("Kidney Disease Classifier")

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

def main():
    clApp = ClientApp()

    # Use a sidebar for navigation
    st.sidebar.write("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Predict", "Train"])

    if page == "Predict":
        st.write("Upload an image for prediction:")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            # Read and encode the uploaded image
            image = uploaded_image.read()
            encoded_image = base64.b64encode(image).decode('utf-8')
            
            # Decode the image and make a prediction
            decodeImage(encoded_image, clApp.filename)
            result = clApp.classifier.predict()
            
            # Display the prediction result
            st.write("Prediction Result:")
            st.write(result[0]["image"])

    elif page == "Train":
        st.write("Training Parameters")

        # Collect training parameters from the user
        AUGMENTATION = st.checkbox("Data Augmentation", value=True)
        IMAGE_SIZE = st.text_input("Image Size (e.g., 224, 224, 3)", "224, 224, 3")
        BATCH_SIZE = st.number_input("Batch Size", 16)
        INCLUDE_TOP = st.checkbox("Include Top Layers", value=False)
        EPOCHS = st.number_input("Number of Epochs", 1)
        CLASSES = st.number_input("Number of Classes", 2)
        WEIGHTS = st.text_input("Model Weights (e.g., imagenet)", "imagenet")
        LEARNING_RATE = st.number_input("Learning Rate", 0.01)

        if st.button("Save Parameters"):
            # Save the parameters to a YAML file
            params = {
                "AUGMENTATION": AUGMENTATION,
                "IMAGE_SIZE": [int(val) for val in IMAGE_SIZE.split(",")],
                "BATCH_SIZE": BATCH_SIZE,
                "INCLUDE_TOP": INCLUDE_TOP,
                "EPOCHS": EPOCHS,
                "CLASSES": CLASSES,
                "WEIGHTS": WEIGHTS,
                "LEARNING_RATE": LEARNING_RATE
            }
            
            if INCLUDE_TOP:
                params["IMAGE_SIZE"] = [224, 224, 3]
            
            params_file_path = "/home/nikhil/Projects/Kidney-Disease-ML-Classifier/params.yaml"
            with open(params_file_path, "w") as yaml_file:
                yaml.dump(params, yaml_file, default_flow_style=False)
            
            st.write("Parameters saved successfully!")

        if st.button("Train Model"):
            # Trigger the training process
            os.system("python main.py")
            st.write("Training done successfully!")

            scores_file_path = "/home/nikhil/Projects/Kidney-Disease-ML-Classifier/scores.json"
            
            config_manager = ConfigurationManager()
            mlflow_uri = config_manager.get_evaluation_config().mlflow_uri
            # Display model scores and mlflow uri if available
            try:
                with open(scores_file_path, "r") as json_file:
                    scores = json.load(json_file)
                    loss = scores.get("loss")
                    accuracy = scores.get("accuracy")
                    st.write("Model Scores:")
                    st.write(f"Loss: {loss:.4f}")
                    st.write(f"Accuracy: {accuracy:.4f}")
                    st.write(f"MflowURI: {mlflow_uri}")
            except FileNotFoundError:
                st.write("Scores file not found. Train the model to generate scores.")

if __name__ == "__main__":
    main()
