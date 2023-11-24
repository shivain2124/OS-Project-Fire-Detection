import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from multiprocessing import Pool
from multiprocessing import cpu_count

def load_and_preprocess_image(args):
    filename, label, directory = args
    try:
        image = cv2.imread(os.path.join(directory, filename))

        # Resize the image to a fixed size (e.g., 128x128 pixels)
        image = cv2.resize(image, (128, 128))
        
        return image, label
    except Exception as e:
        print(f"Error loading image {filename} from {directory}: {e}")
        return None, None

def main():
    # Paths to your dataset folders
    fire_directory = r"/Users/bhyan/Downloads/fire_dataset/fire_images"
    non_fire_directory = r"/Users/bhyan/Downloads/fire_dataset/non_fire_images"

    # Load fire images and assign label 1
    fire_images = [(filename, 1, fire_directory) for filename in os.listdir(fire_directory) if filename.endswith(('.jpg', '.png'))]

    # Load non-fire images and assign label 0
    non_fire_images = [(filename, 0, non_fire_directory) for filename in os.listdir(non_fire_directory) if filename.endswith(('.jpg', '.png'))]

    # Combine the lists
    all_images = fire_images + non_fire_images

    # Use multiprocessing to load and preprocess images
    with Pool() as pool:
        results = pool.map(load_and_preprocess_image, all_images)

    # Print debugging information
    print("Number of results:", len(results))
    print("Results:", results)

    # Remove None values from results
    valid_results = [result for result in results if result[0] is not None]

    if not valid_results:
        print("No valid results. Check the image loading and preprocessing process.")
        exit()

    # Separate the results back into X and y
    X, y = zip(*valid_results)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("Number of samples in X:", len(X))
    print("Number of samples in y:", len(y))

    # Normalize pixel values to be between 0 and 1
    X = X.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y = to_categorical(y, 2)  # 2 is the number of classes (fire and non-fire)

    # Split the dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build the CNN Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Output layer with 2 units (fire and non-fire)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the Model (optional)
    try:
        loss, accuracy = model.evaluate(X_test, y_test)
        print("Test Accuracy:", accuracy)

        # Generate predictions
        y_pred = model.predict(X_test)

        # Convert predictions to labels (0 or 1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        # Create a confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        print("Confusion Matrix:")
        print(cm)

        # Print a classification report for more detailed metrics
        print("Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
    except Exception as e:
        print("Error during evaluation:", e)

    # Process Images and Ring Alarm for Fire Detection
    for i in range(len(X_test)):
        current_image = X_test[i]

        # Perform prediction using your trained model
        predictions = model.predict(np.expand_dims(current_image, axis=0))

        # Check if the prediction indicates fire (you may need to adjust the threshold)
        if predictions[0, 1] > 0.25:
            print("Fire detected in image {}".format(i))

            # Ring the alarm using winsound (adjust the frequency and duration as needed)
            frequency = 2500  # Set frequency to 2500 Hertz
            duration = 1000  # Set duration to 2000 milliseconds (2 seconds)
            #winsound.Beep(frequency, duration)
        else:
            print("No fire detected in image {}".format(i))

    # Display the final evaluation results
    print("Final Test Accuracy:", accuracy)

if __name__ == "__main__":
    main()