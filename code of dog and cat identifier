import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import os

def identify_dog_cat(image_path):
    """
    Identifies whether an image contains a dog or a cat using a pre-trained MobileNetV2 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A message indicating whether a dog, cat, or neither was detected,
             along with the top prediction and its confidence.
        float: The confidence score of the top prediction.
    """
    try:
        # Load the pre-trained MobileNetV2 model
        # 'weights='imagenet'' means it's pre-trained on the ImageNet dataset
        # 'include_top=True' means we include the fully connected layer at the top
        # that is responsible for classification.
        model = MobileNetV2(weights='imagenet', include_top=True)

        # Load the image and resize it to the target size expected by MobileNetV2 (224x224)
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)  # Convert the image to a NumPy array
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension (1, 224, 224, 3)

        # Preprocess the image for the MobileNetV2 model
        # This scales pixel values to the range [-1, 1]
        preprocessed_img = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(preprocessed_img)

        # Decode the predictions to get human-readable labels and probabilities
        # top=3 means we want the top 3 predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        result_message = f"Analyzing image: {os.path.basename(image_path)}\n"
        top_prediction_label = "N/A"
        top_prediction_confidence = 0.0

        dog_breeds = [
            'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih-tzu',
            'basset', 'beagle', 'bloodhound', 'bluetick', 'walker_hound',
            'english_foxhound', 'redbone', 'borzoi', 'irish_wolfhound', 'italian_greyhound',
            'whippet', 'afghan_hound', 'basset_hound', 'boxer', 'rottweiler',
            'german_shepherd', 'doberman', 'miniature_schnauzer', 'giant_schnauzer',
            'standard_schnauzer', 'bernese_mountain_dog', 'newfoundland',
            'great_pyrenees', 'samoyed', 'pomeranian', 'chow', 'husky',
            'malamute', 'siberian_husky', 'dalmatian', 'affenpinscher',
            'basenji', 'pug', 'leonberg', 'great_dane', 'saint_bernard',
            'akita', 'boston_terrier', 'scottish_terrier', 'airedale',
            'irish_terrier', 'welsh_terrier', 'fox_terrier', 'miniature_poodle',
            'standard_poodle', 'toy_poodle', 'labrador_retriever',
            'golden_retriever', 'chesapeake_bay_retriever',
            'flat-coated_retriever', 'curly-coated_retriever',
            'german_shorthaired_pointer', 'vizsla', 'english_setter',
            'irish_setter', 'gordon_setter', 'weimaraner', 'rhodesian_ridgeback',
            'bichon_frise', 'papillon', 'cocker_spaniel', 'english_springer',
            'breton_spaniel', 'clumber_spaniel', 'shetland_sheepdog',
            'collie', 'old_english_sheepdog', 'bouvier_des_flandres',
            'briard', 'entlebucher', 'greater_swiss_mountain_dog',
            'australian_shepherd', 'border_collie', 'komondor', 'kuvasz',
            'belgian_shepherd', 'malinois', 'groenendael', 'tibetan_terrier',
            'lhasa_apso', 'norwegian_elkhound', 'keeshond', 'schipperke',
            'mexican_hairless', 'dingo', 'african_hunting_dog', 'otterhound',
            'komondor', 'kuvasz', 'malamute', 'siberian_husky', 'eskimo_dog',
            'saluki', 'borzoi', 'irish_wolfhound', 'italian_greyhound',
            'whippet', 'ibizan_hound', 'norwegian_elkhound', 'keeshond',
            'schipperke', 'gordon_setter', 'english_setter', 'irish_setter',
            'weimaraner', 'vizsla', 'rhodesian_ridgeback', 'basset_hound',
            'beagle', 'bloodhound', 'bluetick', 'walker_hound',
            'english_foxhound', 'redbone', 'chihuahua', 'japanese_spaniel',
            'maltese_dog', 'pekinese', 'shih-tzu', 'affenpinscher',
            'basenji', 'pug', 'leonberg', 'great_dane', 'saint_bernard',
            'akita', 'boston_terrier', 'scottish_terrier', 'airedale',
            'irish_terrier', 'welsh_terrier', 'fox_terrier', 'miniature_poodle',
            'standard_poodle', 'toy_poodle', 'labrador_retriever',
            'golden_retriever', 'chesapeake_bay_retriever',
            'flat-coated_retriever', 'curly-coated_retriever',
            'german_shorthaired_pointer', 'vizsla', 'english_setter',
            'irish_setter', 'gordon_setter', 'weimaraner', 'rhodesian_ridgeback',
            'bichon_frise', 'papillon', 'cocker_spaniel', 'english_springer',
            'breton_spaniel', 'clumber_spaniel', 'shetland_sheepdog',
            'collie', 'old_english_sheepdog', 'bouvier_des_flandres',
            'briard', 'entlebucher', 'greater_swiss_mountain_dog',
            'australian_shepherd', 'border_collie', 'komondor', 'kuvasz',
            'belgian_shepherd', 'malinois', 'groenendael', 'tibetan_terrier',
            'lhasa_apso', 'norwegian_elkhound', 'keeshond', 'schipperke',
            'mexican_hairless'
        ]

        cat_breeds = [
            'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat',
            'brindled_cat', 'british_shorthair', 'maine_coon', 'ragdoll', 'sphynx',
            'bombay', 'abyssinian', 'birman', 'russian_blue', 'scottish_fold',
            'oriental_shorthair', 'american_shorthair', 'turkish_angora', 'himalayan',
            'chartreux', 'burmese', 'ocicat', 'siberian_cat', 'devon_rex', 'cornish_rex'
        ]

        is_dog = False
        is_cat = False

        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            result_message += f"  Prediction {i+1}: {label} ({score:.2f})\n"
            if i == 0:
                top_prediction_label = label
                top_prediction_confidence = score

            if any(dog_breed in label.lower() for dog_breed in dog_breeds):
                is_dog = True
            if any(cat_breed in label.lower() for cat_breed in cat_breeds):
                is_cat = True

        if is_dog and is_cat:
            result_message += "  Both dog and cat characteristics detected (unlikely, check top prediction).\n"
        elif is_dog:
            result_message += f"  Result: This image likely contains a DOG. Top prediction: {top_prediction_label} ({top_prediction_confidence:.2f})\n"
        elif is_cat:
            result_message += f"  Result: This image likely contains a CAT. Top prediction: {top_prediction_label} ({top_prediction_confidence:.2f})\n"
        else:
            result_message += f"  Result: Neither a clear dog nor cat was detected. Top prediction: {top_prediction_label} ({top_prediction_confidence:.2f})\n"

        # Display the image with the prediction
        plt.imshow(img)
        plt.title(f"Prediction: {top_prediction_label} ({top_prediction_confidence:.2f})")
        plt.axis('off')
        plt.show()

        return result_message, top_prediction_confidence

    except FileNotFoundError:
        return "Error: Image file not found.", 0.0
    except Exception as e:
        return f"An error occurred: {e}", 0.0

if __name__ == "__main__":
    # Create some dummy image files for testing
    # In a real scenario, you would replace these with your actual image paths.
    # For demonstration, we'll download some images.
    print("Downloading sample images for demonstration...")
    try:
        import requests

        def download_image(url, filename):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
            print(f"Downloaded {filename}")

        # Ensure the 'test_images' directory exists
        if not os.path.exists('test_images'):
            os.makedirs('test_images')

        dog_image_url = "https://www.purina.co.uk/sites/default/files/2022-09/Can%20Dogs%20Eat%20Rice%20HERO.jpg"
        cat_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
        not_animal_image_url = "https://cdn.mos.cms.futurecdn.net/4Wn49p29g2mXN8oDq8B6mP-970-80.jpg" # Example of a non-animal image

        dog_image_path = os.path.join('test_images', 'golden_retriever.jpg')
        cat_image_path = os.path.join('test_images', 'siamese_cat.jpg')
        not_animal_image_path = os.path.join('test_images', 'apple.jpg')

        download_image(dog_image_url, dog_image_path)
        download_image(cat_image_url, cat_image_path)
        download_image(not_animal_image_url, not_animal_image_path)

    except ImportError:
        print("Requests library not found. Please install it: pip install requests")
        print("Skipping image download. You will need to provide your own image paths.")
        dog_image_path = 'path/to/your/dog_image.jpg'
        cat_image_path = 'path/to/your/cat_image.jpg'
        not_animal_image_path = 'path/to/your/not_animal_image.jpg'
    except Exception as e:
        print(f"An error occurred during image download: {e}")
        print("Please ensure you have an internet connection and the URLs are valid.")
        dog_image_path = 'path/to/your/dog_image.jpg'
        cat_image_path = 'path/to/your/cat_image.jpg'
        not_animal_image_path = 'path/to/your/not_animal_image.jpg'


    print("\n--- Testing with a Dog Image ---")
    message, confidence = identify_dog_cat(dog_image_path)
    print(message)

    print("\n--- Testing with a Cat Image ---")
    message, confidence = identify_dog_cat(cat_image_path)
    print(message)

    print("\n--- Testing with a Non-Animal Image (e.g., an apple) ---")
    message, confidence = identify_dog_cat(not_animal_image_path)
    print(message)

    print("\n--- Testing with a non-existent image ---")
    message, confidence = identify_dog_cat("non_existent_image.jpg")
    print(message)
