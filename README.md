# Lights, Camera, Inequality: The Story Of Women in Cinema

## Abstract: 

Cinema reflects society, showcasing its values and, often, its inequalities. Historically, women on screen have been sidelined to secondary or stereotypical roles, overshadowed by male characters. But has this improved over time? Are modern films truly more inclusive, or is “feminism” just a buzzword?
This project dives into the representation of women in cinema, exploring whether female roles have increased, improved, or gained prominence. By analyzing over 80,000 films—spanning genres, countries, and decades—we examine trends like the prevalence of female-led narratives, thematic depth, and the proportion of movies passing the Bechdel test. We also investigate whether certain genres or countries excel in portraying women.
Using tools like GPT-2 for textual analysis and machine learning models for classification, this study uncovers insights into the evolution of women’s roles in film. Ultimately, we question what meaningful progress looks like and whether cinema truly mirrors the diversity of its audience.

Research Questions: 
- Are female roles more numerous today than they were in past decades?
- Are female characters more prevalent in certain film genres (drama, comedy, action, etc.) compared to others? And what about across different countries?
- How has the proportion of films passing the Bechdel test evolved over time? And how does it vary across genres and countries?
- What characteristics define a “feminist film” (e.g., themes, characters, etc.)? Do these films have specific markers in their plots or narrative styles?
- How can graph theory be used to analyze the relationships between characters in films, and do films with strong female roles or feminist themes exhibit distinct graph structures (e.g., centrality, clustering, or connectivity) compared to others?

If you want to know the answer to these questions, do not hesitate to click on this link: (https://jbenand.github.io/index)


## Proposed additional datasets :  
The following dataset helps us determine if a movie passes the Bechdel test or not: https://www.kaggle.com/datasets/vinifm/female-representation-in-cinema
https://r-packages.io/datasets/bechdel
This dataset will be used to train a machine learning model able to detect feminist movies through their plot.
We also used a dataset  of movies that were nominated or won the Best Picture Oscar: https://www.kaggle.com/datasets/martinmraz07/oscar-movies/data
This dataset allowed us to determine the number of Bechdel-passing and feminist films that won an Oscar or were nominated.
To define if a movie was feminist, we create our own dataset based on the following lists: 
https://letterboxd.com/brunaleo/list/essential-feminist-films-worldwide/

## Methods: 

Our project is separated into two main parts: Bechel test and feminism movies. In each part, we implement several models to determine if a movie passes the Bechdel Test or if it can be considered as a feminism movie. To train them, we used the previous dataset that contains 1800 movies. (rajouter détails du dataset ? )

### Part 1: Bechdel Test

#### GPT-2 model

GPT-2 is a pre-trained language model that specializes in understanding and generating natural language. It excels in analyzing text and capturing the nuances of meaning within a given context. In this case, GPT-2 was used to analyze movie plot summaries to determine whether a movie passes the Bechdel Test. 

We turned to GPT-2, a pre-trained language model that specializes in generating and understanding natural language. GPT-2 was chosen for its ability to analyze the semantics of movie plot summaries, helping us identify nuanced themes and context that go beyond simple keyword matching. By fine-tuning GPT-2 on a custom dataset of feminist movies, we were able to improve its ability to recognize feminist elements in films, providing a richer and more accurate classification than the Bechdel Test alone.

This method is especially valuable in addressing the limitations of the Bechdel Test, which can miss films with feminist messages that might not meet its criteria. GPT-2 allows us to go beyond surface-level analysis and dive deeper into the language of movie summaries, identifying key themes, character arcs, and narrative structures that reflect feminist ideas.

#### Support Vector Machine (SVM) model

SVM, a supervised machine learning algorithm,  was selected as it excels in handling binary classification tasks using numerical and categorical input features. For the second metric—where female cast proportion is included—SVM provides a straightforward and interpretable model to classify movies with high accuracy.

Once trained, the model can then classify new movies based on the proportion of female actresses in the cast. It assigns movies to one of the two categories: those that are considered to have good representation of women and those that don’t.

### Part 2: Feminism movies

To determine if a movie is feminist or not, we also implement a GPT-2 model. We selected 300 movies that were considered feminist (see link above) and 300 movies that failed the Bechdel Test to constitute our training set. We then use this model to predict if a movie in our initial dataset could be considered as feminist or not. 

## Contributions of each group member: 

Shrinidhi Singaravelan: Implementation of GPT models, data analysis and result plotting, website, and writing.

Marianne Civit-Ardevol: Data analysis for the graph theory section, results plotting, and writing.

Dorah Borgi: Introduction, general analysis and plots of the observations in each section, and writing.

Jacques Bénand : setting up the website, implementation of the SVM model and results plotting

Mathieu :  website and writing. 



