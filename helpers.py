import pandas as pd
import ast

########### Helper functions ###########

def create_datasets(file_source = '', reviews_source = ''):
    # MetaData 

    #### Movie metadata #### DF
    MovieMetadata_df = pd.read_csv(file_source +'/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)

    #### Character metadata #### DF
    CharacterMetadata_df = pd.read_csv(file_source + '/MovieSummaries/character.metadata.tsv', sep='\t', header=None)

    # Define column headers as a list
    MovieMetadata_df_headers = [
        'Wikipedia movie ID',
        'Freebase movie ID',
        'Movie name',
        'Movie release date',
        'Movie box office revenue',
        'Movie runtime',
        'Movie languages (Freebase ID:name tuples)',
        'Movie countries (Freebase ID:name tuples)',
        'Movie genres (Freebase ID:name tuples)'
    ]

    MovieMetadata_df.columns = MovieMetadata_df_headers

    # Define column headers for the second dataset
    CharacterMetadata_df_headers = [
        'Wikipedia movie ID',
        'Freebase movie ID',
        'Movie release date',
        'Character name',
        'Actor date of birth',
        'Actor gender',
        'Actor height (in meters)',
        'Actor ethnicity (Freebase ID)',
        'Actor name',
        'Actor age at movie release',
        'Freebase character/actor map ID',
        'Freebase character ID',
        'Freebase actor ID'
    ]

    CharacterMetadata_df.columns = CharacterMetadata_df_headers

    # Text data 
    names_df = pd.read_csv(file_source+'MovieSummaries/name.clusters.txt', sep="\t", header=None)
    plot_summaries_df = pd.read_csv(file_source+'MovieSummaries/plot_summaries.txt', sep="\t", header=None)
    tvTropes_df = pd.read_csv(file_source+'MovieSummaries/tvtropes.clusters.txt', sep="\t", header=None)

    names_df_headers = [
        'Character Names',
        'Instances'
    ]

    names_df.columns = names_df_headers

    tvTropes_df_headers = [
        'Character Types',
        'Instances'
    ]

    tvTropes_df.columns = tvTropes_df_headers


    plot_summaries_df_headers = [
        'Wikipedia movie ID',
        'Summaries'
    ]

    plot_summaries_df.columns = plot_summaries_df_headers

    reviews = pd.read_csv(reviews_source)

    merged_Movie = MovieMetadata_df.merge(reviews, how='left', left_on = 'Movie name', right_on = 'title')


    # Function to extract the year
    def extract_year(date_str):
        if pd.isna(date_str):  # Check if the value is NaN
            return None  # Return None or a suitable placeholder for NaN
        if len(date_str) == 4:  # Check if it's a four-digit year
            return int(date_str)  # Return the year as an integer
        return pd.to_datetime(date_str, errors='coerce').year  # Convert to datetime and extract year


    # Extract the year without modifying the original column
    MovieMetadata_df['Year'] = MovieMetadata_df['Movie release date'].apply(extract_year)
    #MovieMetadata_df['Year'] = MovieMetadata_df['Year'].apply(lambda x: int(x) if pd.notna(x) else x)
    MovieMetadata_df['Year'] = pd.to_numeric(MovieMetadata_df['Year'], errors='coerce').astype('Int64')


    CharacterMetadata_df['Year'] = CharacterMetadata_df['Movie release date'].apply(extract_year)
    CharacterMetadata_df['Year'] = pd.to_numeric(CharacterMetadata_df['Year'], errors='coerce').astype('Int64')
    MovieMetadata_df_filtered = MovieMetadata_df[['Wikipedia movie ID', 'Movie countries (Freebase ID:name tuples)',
                                              'Movie genres (Freebase ID:name tuples)', 'Country dictionnaire', 'Genre dictionnaire', 
                                              'Movie box office revenue' ]]
    CharacterMetadata_df = pd.merge(CharacterMetadata_df,MovieMetadata_df_filtered, on = 'Wikipedia movie ID', how='left' )


    return MovieMetadata_df, CharacterMetadata_df, names_df, plot_summaries_df, tvTropes_df, merged_Movie

# Function to count the number of movies per country and keep those with >500 movies
def movie_per_country(countries):
    countries_cleaned = []
    for country in countries:
        if len(country) > 2:
            data_dict = ast.literal_eval(country)  # Convert the string to a dictionary
            country_name = list(data_dict.values())[0]  
            countries_cleaned.append(country_name)

    # count the number of occurences of movies for each country
    countries_cleaned = pd.Series(countries_cleaned)
    counted_countries = countries_cleaned.value_counts().sort_index()
    #keep only those that have more than 500 movies
    kept_counted = counted_countries[counted_countries.values>500]
    return kept_counted

# Function to create a merged dataset between bechdel test and movie plots
def bechdel_plots_dataset_creation(bechdel_data, MovieMetadata_df, plot_summaries_df):
    # Keep only the essential elements of the bechdel dataset --> movie title and bechdel test
    bechdel_data_essential = bechdel_data[['title','bt_score']]

    # Keep only bechdel test = 0 or test = 3 (feminist and non-feminist)
    bechdel_data_essential = bechdel_data_essential[bechdel_data_essential['bt_score'].isin([0, 3])]

    # Rename the title to match the movie plots dataset
    bechdel_data_essential.rename(columns = {'title':'Movie name'}, inplace = True)

    # merge the movie plots with their movie names
    movie_ID_names = MovieMetadata_df[['Wikipedia movie ID','Movie name']]
    merged_plots_names = pd.merge(movie_ID_names, plot_summaries_df, on='Wikipedia movie ID', how='inner')

    # merge the feminist movie information with the plots
    merged_bechdel_plot = pd.merge(bechdel_data_essential, merged_plots_names, on='Movie name', how='inner')
    merged_bechdel_plot = merged_bechdel_plot.drop(['Wikipedia movie ID'],axis=1)
    return merged_bechdel_plot

# Function to return the dates filter to only contain the years
def filter_years(MovieMetadata_df):
    years = MovieMetadata_df["Movie release date"].values

    #filter the dates to keep only the years
    filtered_years =[]
    for year in years:
        if isinstance(year, str):
            if len(year) > 4:
                filtered_years.append(year[0:4])
            else:
                filtered_years.append(year)

    # count the number of occurences for each year
    filtered_years = pd.Series(filtered_years)
    return filtered_years
