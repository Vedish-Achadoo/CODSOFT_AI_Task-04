import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#MCU movies from 2014 to 2024 for top 12 most popular characters
data = {
    'title': ['Guardians of the Galaxy', 'Avengers: Age of Ultron', 'Ant-Man', 'Captain America: Civil War', 'Doctor Strange', 
              'Guardians of the Galaxy Vol. 2', 'Spider-Man: Homecoming', 'Thor: Ragnarok', 'Black Panther', 'Avengers: Infinity War', 
              'Ant-Man and The Wasp', 'Captain Marvel', 'Avengers: Endgame', 'Spider-Man: Far From Home', 'Black Widow', 
              'Shang-Chi and the Legend of the Ten Rings', 'Eternals', 'Doctor Strange in the Multiverse of Madness', 'Thor: Love and Thunder', 
              'Black Panther: Wakanda Forever', 'Ant-Man and The Wasp: Quantumania', 'Guardians of the Galaxy Vol. 3', 'The Marvels', 
              'Spider-Man: No Way Home', 'Thor: Love and Thunder', 'Doctor Strange 3', 'Iron Man 4', 'Captain America: New World Order', 
              'Fantastic Four', 'Blade'],
    'superhero': ['Star-Lord', 'Multiple', 'Ant-Man', 'Captain America', 'Doctor Strange', 'Star-Lord', 'Spider-Man', 'Thor', 'Black Panther', 
                  'Multiple', 'Ant-Man', 'Captain Marvel', 'Multiple', 'Spider-Man', 'Black Widow', 'Shang-Chi', 'Eternals', 'Doctor Strange', 
                  'Thor', 'Black Panther', 'Ant-Man', 'Star-Lord', 'Captain Marvel', 'Spider-Man', 'Thor', 'Doctor Strange', 'Iron Man', 
                  'Captain America', 'Fantastic Four', 'Blade'],
    'year': [2014, 2015, 2015, 2016, 2016, 2017, 2017, 2017, 2018, 2018, 2018, 2019, 2019, 2019, 2021, 2021, 2021, 2022, 2022, 2022, 2023, 
             2023, 2024, 2021, 2022, 2023, 2023, 2024, 2024, 2024]
}

df_movies = pd.DataFrame(data)

#Function to get recommendations based on user preferences
def get_recommendations(superhero, year):
    filtered_movies = df_movies[(df_movies['superhero'].str.contains(superhero, case=False, na=False)) &
                                (df_movies['year'] == year)]
    if filtered_movies.empty:
        return None
    return filtered_movies['title'].tolist()

#Main function to interact with the user
def main():
    superhero_list = df_movies['superhero'].unique()
    print("Choose a superhero from the following list:")
    for hero in superhero_list:
        print(hero)
    
    #Loop for selecting of the superhero name
    while True:
        superhero = input("Enter the superhero name: ")
        if superhero in superhero_list:
            break
        else:
            print("Invalid superhero name. Please choose from the list.")
    
    #Loop for selecting the year and satisfaction
    while True:
        while True:
            try:
                year = int(input("Enter the year (2014-2024): "))
                if 2014 <= year <= 2024:
                    break
                else:
                    print("Please enter a year between 2014 and 2024.")
            except ValueError:
                print("Invalid input. Please enter a valid year.")

        recommendations = get_recommendations(superhero, year)
        
        if recommendations:
            movie = random.choice(recommendations)
            print(f"Recommended movie: {movie}")
            satisfied = input("Are you satisfied with this movie? (Yes/No): ").strip().lower()
            if satisfied == 'yes':
                print("Enjoy your movie!")
                break
            else:
                change_choice = input("Would you like to choose a different superhero or a different year? (superhero/year): ").strip().lower()
                if change_choice == 'superhero':
                    print("Choose a superhero from the following list:")
                    for hero in superhero_list:
                        print(hero)
                    while True:
                        superhero = input("Enter the superhero name: ")
                        if superhero in superhero_list:
                            break
                        else:
                            print("Invalid superhero name. Please choose from the list.")
                elif change_choice == 'year':
                    continue
                else:
                    print("Invalid choice. Let's try selecting the year again.")
        else:
            print("There's no movie available to choose on the requested year, choose another year.")

#Collaborative Filtering (Example implementation)
def collaborative_filtering_example():
    #Suggested user ratings
    ratings_data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'movie_title': ['Guardians of the Galaxy', 'Avengers: Age of Ultron', 'Ant-Man', 'Captain America: Civil War', 
                        'Doctor Strange', 'Guardians of the Galaxy Vol. 2', 'Spider-Man: Homecoming', 'Thor: Ragnarok', 
                        'Black Panther', 'Avengers: Infinity War'],
        'rating': [5, 4, 4, 5, 4, 5, 4, 5, 5, 4]
    }

    df_ratings = pd.DataFrame(ratings_data)

    #Create the user-item matrix
    user_item_matrix = df_ratings.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)

    #Computes user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    def get_collaborative_recommendations(user_id, num_recommendations=3):
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
        recommended_movies = df_ratings[df_ratings['user_id'].isin(similar_users)]['movie_title'].unique()
        return recommended_movies.tolist()

    #Example: Get recommendations for user 1
    print("Collaborative Recommendations for User 1:", get_collaborative_recommendations(1))

#Content-based Filtering (Example implementation)
def content_based_filtering_example():
    #Create a combined features column
    df_movies['features'] = df_movies['superhero'] + ' ' + df_movies['title']

    #Creates TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movies['features'])

    #Compute movie similarities
    movie_similarity = cosine_similarity(tfidf_matrix)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=df_movies['title'], columns=df_movies['title'])

    def get_content_based_recommendations(movie_title, num_recommendations=3):
        similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False).index[1:num_recommendations+1]
        return similar_movies.tolist()

    #Example: Get recommendations for 'Guardians of the Galaxy'
    print("Content-based Recommendations for 'Guardians of the Galaxy':", get_content_based_recommendations('Guardians of the Galaxy'))

if __name__ == "__main__":
    main()
    collaborative_filtering_example()
    content_based_filtering_example()

#This is the Fourth task assigned by CodSoft to Vedish. Task has been completed.
#First of all to be able to run the code, you have to open Windows CMD(As Administrator) and run the following: pip install pandas numpy
#It's a simple MCU movies recommendation system that suggests movies to users based on their preferences.
#The recommendation system use techniques such collaborative filtering as well as content-based filtering
#It contains 12 Superhero Names and Movies as from 2014 to 2024