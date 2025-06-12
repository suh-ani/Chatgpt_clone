import requests

# 1. Set your API key directly
API_KEY = "d96764aa182b15600e3e70de8aa05af5d1157fba69e94685eb7e014125c70235"

def search_web(query: str) -> str:
    # 2. SerpAPI endpoint for Google Search
    url = "https://serpapi.com/search"
    
    # 3. Parameters to send in the request
    params = {
        "q": query,            # the search query
        "api_key": API_KEY,    # your SerpAPI key
        "engine": "google"     # tells SerpAPI to use Google search engine
    }

    # 4. Make the HTTP GET request
    resp = requests.get(url, params=params).json()

    # 5. Extract the organic search results
    results = resp.get("organic_results", [])

    # 6. Format and return the top 3 results
    return "\n\n".join(f"{r['title']}: {r['link']}" for r in results[:3])