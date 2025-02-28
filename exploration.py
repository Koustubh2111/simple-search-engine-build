import pandas as pd
import ast
import json

def read_data(filepath, col_names, delimiter = '\t'):
    return pd.read_csv(filepath, delimiter=delimiter, names = col_names)


def join_tables(df1,df2, key):
    return df1.merge(df2, on=key, how = 'left')

def return_val(s):
    return list(ast.literal_eval(s).values())[0]

def format_record(row):

    return {
        'wikipedia_id' : row['WikiID'],
        'movie_name' : row['Movie_name'],
        'release_date' : row['Movie_release_date'],
        'box_office' : row['Movie box office revenue'],
        'runtime' : row['Movie runtime'],
        'summary' : row['PlotSummary']
    }


def dump_json():

    metadata_cols = [
                "WikiID", 
                "Freebase movie ID",
                "Movie_name",
                "Movie_release_date",
                "Movie box office revenue",
                "Movie runtime",
                "Movie_languages_FreebaseID_name_tuples",
                "Movie_countries_FreebaseID_name_tuples",
                "Movie_genres_FreebaseID_name_tuples"
                ]

    df_plot_sum = read_data("./MovieSummaries/plot_summaries.txt", \
                            col_names= ['WikiID', 'PlotSummary'])

    df_movie_meta = read_data("./MovieSummaries/movie_metadata.tsv", col_names=metadata_cols)

    movies = join_tables(df_movie_meta, df_plot_sum, key='WikiID')


    movies_json = [format_record(row) for _, row in movies.iterrows()]

    with open("./movies.json", "w") as f:
        json.dump(movies_json, f, indent=4)


if __name__ == "__main__":
    dump_json()

