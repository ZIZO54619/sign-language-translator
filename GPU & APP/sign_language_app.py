import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import gradio as gr

# Load pre-trained models
arabic_model = load_model('arabic_sign_language_model.h5')
english_model = load_model('english_sl_model_v2.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


# Extract and scale hand landmarks for model input
def extract_and_scale_landmarks(image, img_size=400, padding=50):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return np.zeros(63), image, False

    landmarks = results.multi_hand_landmarks[0]
    handedness = "Right"
    if results.multi_handedness and len(results.multi_handedness) > 0:
        handedness = results.multi_handedness[0].classification[0].label

    # Flip image if left hand is detected
    if handedness == "Left":
        image = cv2.flip(image, 1)
        for landmark in landmarks.landmark:
            landmark.x = 1 - landmark.x

    # Extract landmark coordinates
    points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    hand_width, hand_height = x_max - x_min, y_max - y_min
    if hand_width < 10 or hand_height < 10:
        return np.zeros(63), image, False

    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

    hand_width, hand_height = x_max - x_min, y_max - y_min
    scale = (img_size - 2 * padding) / max(hand_width, hand_height)
    offset_x = (img_size - hand_width * scale) / 2
    offset_y = (img_size - hand_height * scale) / 2

    scaled_points = [
        (int((x - x_min) * scale + offset_x), int((y - y_min) * scale + offset_y))
        for x, y in points
    ]

    normalized_points = []
    for x, y in scaled_points:
        normalized_points.extend([x / img_size, y / img_size, 0.0])

    return np.array(normalized_points), image, True


# Define label mappings for Arabic and English sign language
arabic_labels_alphabets = ['ع', 'ال', 'ا', 'ب', 'ض', 'د', 'ف', 'غ', 'ح', 'ه', 'ج', 'ك', 'خ', 'لا', 'ل', 'م', 'ن', 'ق',
                           'ر', 'ص',
                           'س', 'ش', 'ط', 'ت', 'ة', 'ذ', 'ث', 'و', 'ي', 'ظ', 'ز']
english_labels_alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                            'S', 'T',
                            'U', 'V', 'W', 'X', 'Y', 'Z']
arabic_labels_words = ['مرحبا', 'مع السلامة', 'من فضلك', 'شكرا', 'آسف', 'نعم', 'لا', 'مساعدة', 'حب', 'أسرة', 'صديق',
                       'أم', 'أب', 'أكل', 'شرب', 'نوم', 'ذهب', 'تعال', 'أريد', 'أحتاج', 'بيت', 'مدرسة', 'عمل', 'وقت',
                       'جيد', 'سيء', 'الله', 'سلام', 'دعاء', 'كتاب', 'ماء']
english_labels_words = ['Hello', 'Goodbye', 'Please', 'Thank you', 'Sorry', 'Yes', 'No', 'Help', 'Love', 'Family',
                        'Friend', 'Mother', 'Father', 'Eat', 'Drink', 'Sleep', 'Go', 'Come', 'Want', 'Need', 'Home',
                        'School', 'Work', 'Time', 'Good', 'Bad']

arabic_label_map_alphabets = {i: label for i, label in enumerate(arabic_labels_alphabets)}
arabic_label_map_words = {i: label for i, label in enumerate(arabic_labels_words)}
english_label_map_alphabets = {i: label for i, label in enumerate(english_labels_alphabets)}
english_label_map_words = {i: label for i, label in enumerate(english_labels_words)}


# Processing function for Gradio
def process_frame(frame, language_model, label_type):
    # Frame is in RGB from Gradio webcam, convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Select model and label map based on dropdown selections
    if language_model == "Arabic":
        model = arabic_model
        label_map = arabic_label_map_alphabets if label_type == "Alphabets" else arabic_label_map_words
    else:
        model = english_model
        label_map = english_label_map_alphabets if label_type == "Alphabets" else english_label_map_words

    # Extract landmarks and process frame
    landmarks, frame_processed, hand_detected = extract_and_scale_landmarks(frame_bgr)
    if hand_detected:
        landmarks = landmarks.reshape(1, 63)
        prediction = model.predict(landmarks, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        predicted_letter = label_map.get(predicted_label, "None")
    else:
        predicted_letter = "None"
        confidence = 0.0


    # Convert back to RGB for Gradio display
    frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
    return frame_rgb, predicted_letter


# Button functions for sentence manipulation
def add_to_sentence(prediction, sentence, label_type):
    if prediction != "None":
        if label_type == "Alphabets":
            new_sentence = sentence + prediction
        else:
            new_sentence = sentence + prediction + " "
        print(f"Adding: {prediction}, New sentence: {new_sentence}")
        return new_sentence
    return sentence


def backspace(sentence, label_type):
    if sentence:
        if label_type == "Alphabets":
            return sentence[:-1]
        else:
            words = sentence.split()
            if words:
                return " ".join(words[:-1]) + " " if len(words) > 1 else ""
            return ""
    return sentence


def add_space(sentence):
    return sentence + " "

def update_message(char, msg):
    return msg + char if char else msg


def delete_last(msg):
    return msg[:-1]


with gr.Blocks() as demo:
    gr.Markdown("# SilentTalker")

    with gr.Row():
        language_model = gr.Dropdown(["Arabic", "English"], label="Language Model", value="Arabic")
        label_type = gr.Dropdown(["Alphabets", "Words"], label="Label Type", value="Alphabets")

    with gr.Row():
        webcam = gr.Image(label="Webcam", type="numpy")
        with gr.Column():
            prediction = gr.Textbox(label="Current Prediction")
            sentence_display = gr.Textbox(label="Sentence")
            with gr.Row():
                add_button = gr.Button("Add")
                space_button = gr.Button("SPACE")
                delete_button = gr.Button("Delete")

    gr.Markdown("Developed with ❤️ by [SilenTalker Team](https://famous-cobbler-46de1c.netlify.app/)")

    # Stream to update webcam and prediction
    webcam.stream(
        fn=process_frame,
        inputs=[webcam, language_model, label_type],
        outputs=[webcam, prediction]  # Update webcam display and prediction
    )

    # Add button: Update sentence_state and sentence_display
    add_button.click(update_message, [prediction, sentence_display], sentence_display)
    delete_button.click(delete_last, sentence_display, sentence_display)
    space_button.click(lambda msg: msg + " ", sentence_display, sentence_display)

demo.launch(share=True)


