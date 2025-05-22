import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Function to preprocess image
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

# Load the trained model
model = load_model("A:\MTECH(Data Science)\DataSet\Apple\model.h5")  # Update with your model path

# Main function for the web app
def main():
    st.title("Leaf Image Classifier")

    # File upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Add a button for prediction
        if st.button('Predict'):
            # Preprocess the uploaded image
            x = preprocess_image(uploaded_file)

            # Perform the prediction using the loaded model
            predicted_label = model.predict(x)  # Assuming model.predict() returns class probabilities

            # Print the predicted label
            labels = {0: 'Apple_black_rot', 1: 'Apple_cedar_rust', 2: 'Apple_scab'}
            predicted_label = labels[np.argmax(predicted_label)]
            st.write('Predicted Image Leaf is:', predicted_label)

            # Print solution based on predicted label
            if predicted_label == 'Apple_black_rot':
                st.write('Solution of Apple_black_rot:')
                st.write('Task1')
            elif predicted_label == 'Apple_cedar_rust':
                st.write('Solution of Apple_cedar_rust:')
                st.write('Task2')
            else:
                st.write('Solution of Apple_scab:')
                st.write('Task3')

# Run the app
if __name__ == "__main__":
    main()
