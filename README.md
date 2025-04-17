# IMDB Movie Review Sentiment Analyzer

![IMDB Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/IMDB_Logo_2016.svg/320px-IMDB_Logo_2016.svg.png)

## Live Demo
[Try the app here](https://moviereviewrnn-futwqffpmhtjadxh8pd5zg.streamlit.app/)

## Project Overview
This project is a web application that analyzes movie reviews and predicts whether they express positive or negative sentiment. The application uses a Recurrent Neural Network (RNN) trained on the IMDB dataset to make these predictions.

## Features
- User-friendly web interface built with Streamlit
- Real-time sentiment analysis of movie reviews
- Confidence score for each prediction
- Responsive design
- Visually appealing UI with gradient backgrounds and emoji indicators

## Technology Stack
- **Python**: Core programming language
- **TensorFlow/Keras**: Used for building and loading the RNN model
- **Streamlit**: Web application framework
- **IMDB Dataset**: Used for training the model and providing vocabulary

## How It Works
1. The app loads a pre-trained SimpleRNN model that was trained on the IMDB dataset
2. When a user submits a review, the text is preprocessed (tokenized, encoded, and padded)
3. The model predicts whether the sentiment is positive or negative
4. Results are displayed with appropriate styling and confidence score

## Technical Challenges & Solutions

### Version Compatibility Issues
One of the main challenges faced was compatibility issues between different versions of TensorFlow/Keras and the saved model format. This was resolved by implementing a custom model loader that handles parameter differences:

```python
def load_model_with_custom_objects():
    # Define a custom SimpleRNN class that accepts and ignores 'time_major'
    class CustomSimpleRNN(keras.layers.SimpleRNN):
        def __init__(self, *args, **kwargs):
            # Remove the problematic parameter if it exists
            if 'time_major' in kwargs:
                kwargs.pop('time_major')
            super().__init__(*args, **kwargs)
    
    # Load the model with the custom objects
    model = keras.models.load_model(
        'simple_rnn_imdb.h5',
        custom_objects={'SimpleRNN': CustomSimpleRNN}
    )
    return model
```

### Streamlit Configuration
Another challenge was understanding Streamlit's specific requirements, such as the need to call `st.set_page_config()` as the very first Streamlit command.

### UI/UX Design
Creating an intuitive and visually appealing interface required careful CSS styling and layout design. The app uses custom CSS to create a professional look while ensuring good text contrast and readability.

## What I Learned

Throughout this project, I gained valuable experience in:

1. **Deep Learning Implementation**: Applying RNN models for natural language processing tasks
2. **Model Deployment**: Moving from a trained model to a live web application
3. **Handling Compatibility Issues**: Solving version conflicts between TensorFlow/Keras releases
4. **Frontend Development**: Building user interfaces with Streamlit and custom CSS
5. **Error Handling**: Implementing robust error handling for better user experience
6. **Text Processing**: Working with text data, including tokenization and sequence padding
7. **Version Management**: Understanding the importance of version compatibility in ML deployments
8. **UI/UX Design**: Creating an intuitive and visually appealing user interface
9. **Cloud Deployment**: Deploying the application to Streamlit Cloud

## Future Improvements
- Add support for analyzing reviews in multiple languages
- Implement a more sophisticated model (e.g., LSTM or Transformer-based)
- Include feature for explaining model decisions
- Add batch processing for multiple reviews
- Extend to other types of sentiment analysis beyond binary classification

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run main.py
```

## Requirements
```
streamlit>=1.25.0
tensorflow>=2.10.0
numpy>=1.21.0
```

## Acknowledgments
- The IMDB dataset
- TensorFlow and Keras documentation
- Streamlit community for their excellent examples and documentation
