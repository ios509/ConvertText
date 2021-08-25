"Convert Text into Numerical Data using Python"


"I will start this task by importing the CountVectorizer class from the Scikit-learn library in Python"
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

"Now I will store some texts into a Python list"
text = ["Python", "Css", "Java","HTML"]

"Now I will fit the list into the CountVectorizer function to convert the list of texts into numerical data:"
vect.fit(text)

"Now let’s convert it into an array of numerical data"
train = vect.transform(text)
train.toarray()

"As we have now converted the textual data into numerical data we can also present it in the form of a pandas DataFrame"
import pandas as pd
data = pd.DataFrame(train.toarray(), columns=vect.get_feature_names())

data