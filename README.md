# 📰 Fake News Detection Using Naive Bayes

This project aims to build a machine learning-based web application for detecting fake news articles using **Naive Bayes** classification. The model has been trained on a labeled dataset containing real and fake news, and the web app is built using **Streamlit** to provide an easy-to-use interface for users to check the authenticity of news articles.

## # Project Overview
- **Objective:** Classify news articles as *real* or *fake* based on textual content.  
- **Machine Learning Model:** Naive Bayes (MultinomialNB)  
- **Web Framework:** Streamlit  
- **Dataset:** [Fake News Dataset from Kaggle](https://www.kaggle.com/c/fake-news/data)  
- **Deployment:** Google Colab (for training) & Streamlit web app  

## # Features
- 📝 **Text Preprocessing:**  
  - Tokenization, stopword removal, and TF-IDF vectorization  
- 🧠 **Machine Learning Model:**  
  - Trained using the Naive Bayes classifier for effective text-based classification  
- 📊 **Data Visualization:**  
  - Insights into word distributions and class imbalances  
- 🌐 **User-Friendly Web App:**  
  - Users can input news text and receive a prediction (Real or Fake)  
- 📈 **Performance Evaluation:**  
  - Accuracy, precision, recall, and F1-score metrics  

## # Installation
Follow these steps to set up the project locally:  

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git  
cd fake-news-detection  

# Create a virtual environment (optional but recommended)
python -m venv fake_news_env  
source fake_news_env/bin/activate  # On Windows: fake_news_env\Scripts\activate  

# Install dependencies
pip install -r requirements.txt  

# Run the Streamlit app
streamlit run app.py  
# How to Use
Open the web app using the Streamlit link after running streamlit run app.py.
Paste any news article text into the input field.
Click the "Predict" button to check if the news is Real or Fake.
View the result and confidence score provided by the Naive Bayes model.
# Project Structure
📂 fake-news-detection  
│-- 📂 data              # Dataset files  
│-- 📂 models            # Trained model files  
│-- 📂 notebooks         # Jupyter notebooks for analysis  
│-- app.py               # Streamlit web app script  
│-- preprocess.py        # Text preprocessing functions  
│-- train_model.py       # Model training script  
│-- requirements.txt     # Required dependencies  
│-- README.md            # Project documentation  
# Future Improvements
Implementing more advanced models like SVM, Logistic Regression, or Transformer-based models.
Deploying the app on cloud platforms (e.g., AWS, Heroku).
Adding real-time news scraping and analysis features.

# Contributors
Your Name – Chetan Shakya
