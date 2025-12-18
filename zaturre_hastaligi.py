from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np 
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# Değişken tanımlamaları
DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
CLASS_MODE = "binary"

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.8, 1.2],
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="validation",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)

class_names = list(train_generator.class_indices.keys()) 
images, labels = next(train_generator) 

plt.figure(figsize=(10, 4)) 
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    ax.imshow(images[i])
    ax.set_title(class_names[int(labels[i])]) 
    ax.axis("off")

plt.tight_layout()
plt.show()

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE,3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation= "relu")(x)
x = Dropout(0.5)(x)
pred = Dense(1,activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=pred)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
]

print("Model Summary:")
print(model.summary())

# Tahmin aşaması
pred_probs = model.predict(test_gen, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).ravel()
true_labels = test_gen.classes

# Karmaşıklık Matrisi Görselleştirme
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)


fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap="Blues", colorbar=False, ax=ax)
plt.title("Confusion Matrix")
plt.show()