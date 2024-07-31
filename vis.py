import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

model = tf.keras.models.load_model('/content/drive/My Drive/checkpoints/cp-0015.h5') 

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = load_img(img_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

test_img_path = "/content/HCM/Sick/Directory_22/series0001-Body/img0001--9.jpg" 

# Load and preprocess the image
img_array = load_and_preprocess_image(test_img_path)
img = img_array[0] 

img_rgb = np.repeat(img, 3, axis=-1)

def predict_fn(images):
    images_gray = np.mean(images, axis=-1, keepdims=True)
    preds = model.predict(images_gray)
    return preds

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(img_rgb.astype('double'), predict_fn, top_labels=2, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Grayscale Image')
plt.imshow(np.squeeze(img_array[0]), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('LIME Explanation')
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.axis('off')

plt.tight_layout()
plt.show()

