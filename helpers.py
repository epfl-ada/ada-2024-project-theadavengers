import pandas as pd


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

    return MovieMetadata_df, CharacterMetadata_df, names_df, plot_summaries_df, tvTropes_df, merged_Movie