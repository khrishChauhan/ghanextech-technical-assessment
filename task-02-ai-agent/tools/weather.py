"""Live Weather tool utilizing Open-Meteo Geocoding and Forecast REST APIs."""

from typing import Any, Dict
import requests

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather(city: str) -> Dict[str, Any]:
    """Fetch live real-time weather conditions for a given city.

    Args:
        city: The name of the city (e.g. 'Tokyo', 'London', 'Accra').

    Returns:
        Structured dict with keys: {city, country, temperature_c, windspeed, weather_condition}.

    Raises:
        ValueError: If city is not found or empty.
        RuntimeError: If external network request fails.
    """
    if not city or not city.strip():
        raise ValueError("City name cannot be empty.")

    clean_city = city.strip()
    headers = {"User-Agent": "GhanexTech-AIAgent/1.0"}

    # 1. Geocode city name to coordinates
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_city}&count=1"
        geo_resp = requests.get(geo_url, headers=headers, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error querying geocoding service: {e}")

    results = geo_data.get("results")
    if not results:
        raise ValueError(f"Could not find coordinates for city: '{clean_city}'. Please check the city name.")

    best_match = results[0]
    lat = best_match["latitude"]
    lon = best_match["longitude"]
    resolved_city = best_match.get("name", clean_city)
    country = best_match.get("country", "Unknown")

    # 2. Fetch current weather from Open-Meteo
    try:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        )
        w_resp = requests.get(weather_url, headers=headers, timeout=10)
        w_resp.raise_for_status()
        w_data = w_resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error querying Open-Meteo weather service: {e}")

    current = w_data.get("current_weather", {})
    temp_c = current.get("temperature")
    wind = current.get("windspeed")
    code = current.get("weathercode", 0)
    condition = WMO_WEATHER_CODES.get(code, "Unknown conditions")

    return {
        "city": resolved_city,
        "country": country,
        "temperature_c": temp_c,
        "windspeed": wind,
        "weather_condition": condition,
    }


# Standardized tool definition dictionary
TOOL_DEFINITION: Dict[str, Any] = {
    "name": "get_weather",
    "description": (
        "Get live real-time weather information for any city worldwide. "
        "Returns current temperature in Celsius, windspeed, and weather conditions (e.g. Clear, Overcast, Rain)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name to retrieve weather for, e.g. 'Tokyo', 'London', 'Nairobi'",
            }
        },
        "required": ["city"],
    },
    "function": get_weather,
}
