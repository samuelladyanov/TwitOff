import numpy as np
from sklearn.linear_model import LogisticRegression
from .twitter import vectorize_tweet
from .models import User


def predict_user(user0_username, user1_username, hypo_tweet_text):

    # Query the two users from the DB
    user0 = User.query.filter(User.username == user0_username).one()
    user1 = User.query.filter(User.username == user1_username).one()


    # Get the word embeddings
    user0_vects = np.array([tweet.vect for tweet in user0.tweets])
    user1_vects = np.array([tweet.vect for tweet in user1.tweets])

    # Combine vectorizations into big X mattrix
    X = np.vstack([user0_vects, user1_vects])

    # Create 0s and 1s to generate a y vector (0s top, 1s bottom)

    y = np.concatenate([np.zeros(len(user0.tweets)), 
                        np.ones(len(user1.tweets))])

    # Train Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X,y)

    # Get the word embedding for our hypthetical tweet
    # Make sure embedding is 2D
    hypo_tweet_vect = np.array([vectorize_tweet(hypo_tweet_text)])

    # Generate Prediction
    prediction = log_reg.predict(hypo_tweet_vect)

    # Return just the integer value in the whole array
    return prediction[0]