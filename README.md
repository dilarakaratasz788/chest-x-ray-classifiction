🫁 Chest X-Ray Classification with Deep Learning
This project trains a deep learning–based image classification model using TensorFlow and Keras to accurately classify chest X-ray images.
A pre-trained DenseNet121 architecture is used to distinguish between two classes (e.g., Normal and Pneumonia).
🎯 Project Objective
The goal of this project is to develop a binary image classification system capable of identifying medical conditions from chest X-ray images by leveraging transfer learning and convolutional neural networks.
The dataset used in this project was obtained from a publicly available dataset on Kaggle and is used for educational and research purposes only.
🚀 Key Technical Features
Transfer Learning:
A DenseNet121 model pre-trained on ImageNet is used as a feature extractor to improve performance and reduce training time.
Data Augmentation:
To improve model generalization, random horizontal flipping, rotation, brightness adjustments, and rescaling are applied during training.
Optimization Techniques:
Training stability and performance are enhanced using EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
Efficient Training Pipeline:
Keras ImageDataGenerator is used to efficiently load and preprocess image data during training and validation.
📊 Training & Evaluation Results
The trained model is evaluated on a separate test dataset.
Model performance is analyzed using accuracy and a confusion matrix to better understand classification behavior.
Accuracy, loss curves, and additional evaluation metrics can be added here in future updates.
📂 Model & File Sharing
The trained model (.h5 format) and all related project files can be accessed via the link below:
🔗 [Insert Google Drive or GitHub Release Link Here]
🛠️ Setup & Usage
To run this project on your local machine:
Clone or download the repository
Install the required dependencies:
pip install tensorflow numpy matplotlib scikit-learn
Start the training process:
python train.py
