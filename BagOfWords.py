from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

# Let's make a function to accomplish what we did above, since we'll be calling this function a lot of times
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than
    # searching a list, so we'll convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
#     print(len(meaningful_words))  # just testing
    #
    # 6. Join the words back into one string separated by space,
    # and return the result. We do this instead of returning the list because
    # returning this will make it easier to use with our Bag of Words model.
    return( " ".join( meaningful_words ))

if __name__ == "__main__":

    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    # Reading from the training data file.
    # Here, "header=0" indicates that the first line of the file contains column names,
    # "delimiter=\t" indicates that the fields are separated by tabs, and
    # quoting=3 tells Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
    print(train.shape) #expected (rows, cols) value of (25000, 3)
    print(train.columns.values)

    print(train["review"][0])  # an example of one of the original reviews, with html tags
    example1 = BeautifulSoup(train["review"][0], "html.parser")
    print(example1.get_text())  #html tags removed



    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]",  # The pattern to search for
                          " ",  # The pattern to replace it with
                          example1.get_text())  # The text to search
    print(letters_only)

    lower_case = letters_only.lower()  # Convert to lower case
    words = lower_case.split()  # Split into words to get a list
    print(words)

    #Using nltk to remove stopwords (common words like "a", "and", that don't add much to meaning)
    print(stopwords.words("english"))

    # Remove stop words from "words"
    words_minus_stopwords = [w for w in words if not w in stopwords.words("english")]
    print(words_minus_stopwords)

    # Comparing the length of original list of words vs. list of words excluding stopwords
    print(len(words))
    print(len(words_minus_stopwords))

    # Confirming that there are 437 - 219 = 218 stopwords that have been removed from the original list of words
    diff = [a for a in words + words_minus_stopwords if (a not in words) or (a not in words_minus_stopwords)]
    print(len(diff))  # this value should equal 437-219
    print(437 - 219)

    # Now we'll do what we did above by calling a custom function we wrote above.
    # Testing our function review_to_words on a single review text that has been
    # html parsed, lowercased, non-alphabets removed, stop words removed
    clean_review = review_to_words(train["review"][0])
    print(clean_review)

    # Now let's go thru the entire training set of reviews and clean up all of them at once
    # This might take a few minutes
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    for i in range(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message for user friendliness
        if ((i + 1) % 1000 == 0):
            print("Review {} of {}...".format(i + 1, num_reviews), end="")
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_train_reviews.append(review_to_words(train["review"][i]))
    print("Done!")



    # Now that we have a clean list of all 25,000 reviews, create Bag of Words
    print("Creating the bag of words...", end="")

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    # Note that CountVectorizer comes with its own options to automatically do
    # preprocessing, tokenization, and stop word removal -- for each of these,
    # instead of specifying "None", we could have used a built-in method
    # or specified our own function to use.
    # See the function documentation for more details
    # (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    # However, we wanted to write our own function for data cleaning in this tutorial
    # to show you how it's done step by step.
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)


    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()
    print("Done.")

    print(train_data_features.shape)  # expected (rows, cols) value of (25000, 5000)
    print(vectorizer)

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print(vocab)
    print(len(vocab))  # expected 5000

    import numpy as np

    # Sum up the counts of each vocabulary word
    # Add up all the rows. We'll be left with a single array consisting of 5,000 elements,
    # with each element denoting the count of each corresponding word in the 5,000-word vocabulary.
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print(count, tag)

    # %%timeit -r 1
    print("Training the random forest...")
    from sklearn.ensemble import RandomForestClassifier

    # Initialize a Random Forest classifier with 100 trees
    # 100 is just a default value we chose. More trees may or may not perform better,
    # but will certainly take longer to run.
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_data_features, train["sentiment"])
    print("Done!")

    # Now that the random forest model is trained on the training data,
    # use the model to predict the sentiment label on the test data
    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                       quoting=3)

    # Verify that there are 25,000 rows and 2 columns
    print("(rows, columns): {}".format(test.shape))

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0, num_reviews):
        if ((i + 1) % 1000 == 0):
            print("Review {} of {}...".format(i + 1, num_reviews), end="")
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print("\nMaking prediction on test data...")
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    print("Creating a pandas dataframe with \"id\" and \"sentiment\" columns...")
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    print("Outputting the prediction file to Bag_of_Words_model.csv...")
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
    print("Done!")

