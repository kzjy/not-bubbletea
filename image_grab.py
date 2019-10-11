from google_images_download import google_images_download as dl

response = dl.googleimagesdownload()
search_queries = ['coffee -bubbletea', 'milk -bubbletea', 'beer -bubbletea', 'softdrink -bubbletea', 'juice -bubbletea']
file_format = ['jpg']

def download(query, fformat):
    arguments = {"keywords": query,
                 "format": fformat,
                 "limit": 250,
                 "aspect_ratio": "tall",
                 "chromedriver": r"D:\Projects\notbubbletea\scaper\chromedriver.exe",
                 "size": "medium",
                 "prefix": "coffee"}
    
    try:
        response.download(arguments)
    except FileNotFoundError:
        print("error")


for query in search_queries:
    for fformat in file_format:
        download(query, fformat)
        print()