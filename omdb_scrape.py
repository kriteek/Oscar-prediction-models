import pandas as pd
import requests
import time
from tqdm import tqdm

API_KEY = 'YOUR_API_KEY'  # Your OMDb key
movies_df = pd.read_csv('filtered_data.csv')
movies_df.rename(columns=lambda x: x.lower().strip(), inplace=True)

results = []

for _, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
    title = row['film']
    year = row.get('year_film')
    url = f"http://www.omdbapi.com/?t={title}&y={year}&apikey={API_KEY}"

    # else:  # If year missing, search without year
    #     url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"

    try:
        res = requests.get(url)
        data = res.json()

        imdb_rating = None
        rotten_rating = None
        imdb_error = None
        rt_error = None

        if data['Response'] == 'True':
            imdb_rating = data.get('imdbRating', None)
            if imdb_rating in [None, "N/A"]:
                imdb_error = "not found"

            for rating in data.get('Ratings', []):
                if rating['Source'] == 'Rotten Tomatoes':
                    rotten_rating = rating['Value']
                    break
            if rotten_rating is None:
                rt_error = "not found"
        else:
            imdb_error = "not found"
            rt_error = "not found"

        results.append({
            'movie name': title,
            'year': year,
            'imdb_rating': imdb_rating,
            'rotten_tomatoes': rotten_rating,
            'imdb_error': imdb_error,
            'rt_error': rt_error
        })

    except Exception as e:
        results.append({
            'movie name': title,
            'year': year,
            'imdb_rating': None,
            'rotten_tomatoes': None,
            'imdb_error': str(e),
            'rt_error': str(e)
        })

    time.sleep(0.2)

# Save the results
ratings_df = pd.DataFrame(results)
ratings_df.to_csv('22to25.csv', index=False)
print("✅ Ratings scraped and saved to 22to25.csv")

