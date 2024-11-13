
**Power of film as a cultural mirror**  

#### Quickstart

Start by installing the following files needed for our project:

Install base files : https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz

Install files for feminism movement data from the git repo : https://github.com/fivethirtyeight/data/blob/master/bechdel/movies.csv

Install other files :


```python
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n adaverngers python=3.9
conda activate adavengers

# install requirements
pip install -r requirements.txt
```
<!--StartFragment-->

**Abstract:** 

Our project delves into the influence of major historical events of the second part - WW2, The feminist movement and the space exploration era – on global movies’ trends. Our goal would be to reveal how these different events shape the movie industry, more specifically the different genre popularities, character developments and narrative structures using different tools such as graph theory to reveal relationships, machine learning and natural language processing. We would like to identify different patterns of cultural influences and genre trend evolution embedded in movies over time. We shall also use data analysis to analyze plot summaries, genres, character details and themes across movies. In the scope, we wish to tell a compelling story about how the movie industry is both reflected and influenced by collective consciousness, using historical events as a foundation to trace shifts in public sentiment, cultural values, and artistic expression over time.

**Research Questions:**

- How have major historical events influenced cinema ? 

- Can we link movie plots to the political environment (e.g. inter-character relationships link to US-USSR relationships) ?

- In what ways do movies reflect public sentiment and cultural attitudes towards these events ?

- On the contrary, can fluctuation in movie trends predict a form of societal shift?

- Through graph theory, can we find which are the strongest links between movies and historical events and which part of these events is actually the most represented in movies ?

**Proposed additional datasets :** 

- We would like to get an idea of the viewers’ opinion on the movies, the dataset we will use is : <https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies>

This dataset contains a large number of movies, in order to use it efficiently, we will merge it to the CMU dataset and use only the corresponding movies. It also contains information on the revenue of the movies which would fill the missing values in our original dataset.

- The following dataset helps us define a feminist movie: <https://www.kaggle.com/datasets/vinifm/female-representation-in-cinema>

<https://r-packages.io/datasets/bechdel>

This dataset will be used to train a machine learning model able to detect feminist movies through their plot.

**Methods:** 

Most of our work will be split into 3 tasks: 

1. **Defining the scope of analysis**

How is a movie linked to a historical event ? For example, what makes a movie “feminist” ? This will allow us to separate the target group of movies and do further research on them. 

- 3 methods : 

  - ML on plot descriptions: For the ML method, supervised (training set required) and unsupervised techniques (clustering e.g. linked to an event) can be used, allowing us to predict which of our movies correspond to the event. 

  - Simple word search or genre lookup: straightforward techniques that will be implemented with the use of pandas and string comparisons. We will do brainstorming and research in order to decide on the words and genre linked to the event.

 

2. **Analyzing the selected movies through time, popularity, different definitions of the event and public sentiment**

In this section, we aim to explore how movies are connected to historical events and identify observable trends, which will allow us to draw conclusions and address our research questions.

Our initial approach involves visualizing the data with time-based boxplots. This will help us examine whether the rise of certain movies aligns with specific historical events, enabling us to draw preliminary conclusions.

Next, by analyzing changes in trends over time with targeted keyword searches, we can identify the most significant terms. This will reveal which aspects of historical events were most frequently represented in movies, offering insight into public sentiment and the general mood surrounding these films. To reinforce these findings, we’ll also analyze movie reviews to understand the emotions they evoke, the reactions they spark, and the reasons behind them.

Additionally, it would be valuable to investigate whether different events and movies are interconnected and uncover the underlying factors that create these connections.

3. **Using graph theory to link everything together and getting a visual feedback of the interconnections**

Finally, using graph theory will be a great way of summarizing all of our findings and getting visual feedback on the project research. The nodes will represent movies and historical events whereas the edges will represent the different links between an event and a movie. Furthermore, we can use different sized edges to show a stronger connection.

**Proposed timeline:**


### _Step 0 : Data cleaning and exploration + labeling_ 

- Mainly done for milestone 2. Process and clean movie data for consistency. Remove duplicates, handle missing data, and standardize genres or keywords. Data exploration should be with all extra datasets that we added. Explore the number of movies and its box office in different chosen themes. Make training sets for ML models training (especially for space race movies) 


### _Step 1 : Defining the target movie groups (Week 1)_ 

- Classify all movies per genre and theme (Space, WW2 and/or Feminism for now) using various techniques, as described previously (step 1 of methods). 


### _Step 2 : Data Analysis (Week 2-3)_

- ### Conduct initial exploratory analysis on genre distribution over time, focusing on spikes around key historical events. Apply sentiment analysis to plot summaries, looking for shifts in tone across different periods (e.g., positive vs. negative sentiment before and after major wars). Keywords frequency analysis to find movie genre trends. 

* Use **time series analysis** to measure the correlations between historical events or movements and genre popularity or sentiment changes and study the frequency of movie releases in selected genres and the views of the audience. 

* Use **Diversity analysis** to track diversity trends in films, such as the number of movies with female or minority leads, and correlate these with social movements.

* Use **Topic modeling** to identify recurring themes in movie plot summaries and analyze how these topics shift over time.


### _Step 3 : Visualization (Week 4)_

- Develop visualizations and graph theory to represent findings across events timeline. 

_Step 4 : Compilation of All Our Analysis (Week 5)_

**Organization within the team:**

Shrinidhi & Marianne : Feminism movement 

Dorah & Jacques : Space exploration

Mathieu : WW2

<!--EndFragment-->





