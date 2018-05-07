from app import app
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask

# read the csv file

df = pd.read_csv('database.csv', sep='\t', low_memory=False)
df['movie_title'] = df['movie_title'].apply(lambda x: x.strip())
df = df.fillna('')

# Build correspondence tables between the dataframe's indices and the titles/IDs
df = df.reset_index()
indices_t = pd.Series(df.index, index=df['movie_title'])
indices_id = pd.Series(df.index, index=df['movie_id'])


# Initoalize the Vectorizer
count = CountVectorizer()

# Fit+Transform the matrices for each feature we'll use to generate similarity scores

count_matrix1 = count.fit_transform(df['genres'])
cosine_sim1 = cosine_similarity(count_matrix1, count_matrix1)

count_matrix2 = count.fit_transform(df['plot_keywords'])
cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)

count_matrix3 = count.fit_transform(df['director_name'])
cosine_sim3 = cosine_similarity(count_matrix3, count_matrix3)

count_matrix4 = count.fit_transform(df['actor_1_name'])
cosine_sim4 = cosine_similarity(count_matrix4, count_matrix4)

count_matrix5 = count.fit_transform(df['actor_2_name'])
cosine_sim5 = cosine_similarity(count_matrix5, count_matrix5)

count_matrix6 = count.fit_transform(df['actor_3_name'])
cosine_sim6 = cosine_similarity(count_matrix6, count_matrix6)

# Define the function that takes a movie title/ID as input and generates a list of recommendations
def get_recommendations(title):
    # This part should determine if the input is a title or an ID (quand je teste la fonction get_recommendations avec un ID comme '0499549'
    # la fonction donne les films recommandés, mais si je fais l'appel depuis l'URL "http://localhost:5000/todo/rec/0499549" par exemple, on obtient une erreur et on a Movie not found
    try:
        try:
            id=int(title)
            index = indices_id[id]
        except:

            index = indices_t[title]
        #Build lists containing the similarity scores for each feature
        sim_scores1 = list(enumerate(cosine_sim1[index]))
        sim_scores2 = list(enumerate(cosine_sim2[index]))
        sim_scores3 = list(enumerate(cosine_sim3[index]))
        sim_scores4 = list(enumerate(cosine_sim4[index]))
        sim_scores5 = list(enumerate(cosine_sim5[index]))
        sim_scores6 = list(enumerate(cosine_sim6[index]))
        # Calculate the final similarity score by adding up the sim_scores and weighing them by order of importance
        sim_scores = [
            score1[1] + 0.75 * score2[1] + 0.5 * score3[1] + 0.1 * score4[1] + 0.1 * score5[1] + 0.1 * score6[1] for
            score1, score2, score3, score4, score5, score6 in
            zip(sim_scores1, sim_scores2, sim_scores3, sim_scores4, sim_scores5, sim_scores6)]
        weighted_scores = pd.Series(sim_scores)
        weighted_scores = weighted_scores.sort_values(ascending=False)

        # Select the top 20 movies with the highest similarity scores
        weighted_scores = weighted_scores[1:20]
        movie_indices = [i for i in weighted_scores.index]

        # Add the IMDB score of the movie to its score then sort them by highest score, this way we get the top ranked movies
        final_scores = weighted_scores + df['imdb_score'].iloc[movie_indices]
        final_scores = final_scores.sort_values(ascending=False)

        # Select the final 5 Movies the function will return as a list
        final_scores = final_scores[0:5]
        movie_indices = [i for i in final_scores.index]
        return df[['movie_title']].iloc[movie_indices].values.tolist()
    except:
        return 'movie not found'







