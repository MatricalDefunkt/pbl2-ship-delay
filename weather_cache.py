import json
import os
import time
import math
import threading
import datetime
from typing import Dict, Any, Tuple, Optional, List


class WeatherCache:
    """
    Cache for weather data to minimize API calls to Open-Meteo.
    Features:
    - In-memory cache with file persistence
    - Coordinate-based lookup with tolerance
    - Different expiration times for current and forecast data
    - Reuse forecast data for current weather when possible and vice versa
    - Automatic cleanup of expired entries
    - Cache statistics
    """

    # Constants for cache configuration
    COORD_PRECISION = (
        2  # Round coordinates to 2 decimal places (approx 1.1 km precision)
    )
    CURRENT_WEATHER_TTL = 3600  # 1 hour expiration for current weather
    FORECAST_WEATHER_TTL = 3 * 3600  # 3 hours expiration for forecast weather
    CACHE_FILE = "weather_cache.json"  # File to persist cache
    CLEANUP_INTERVAL = 15 * 60  # Clean up every 15 minutes

    def __init__(self):
        """Initialize the weather cache."""
        self.current_cache: Dict[str, Dict] = {}  # Cache for current weather
        self.forecast_cache: Dict[str, Dict] = {}  # Cache for forecast weather
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cleanup = time.time()
        self.lock = threading.RLock()  # Reentrant lock for thread safety

        # Load cache from disk if available
        self._load_cache()

        # Start the cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        self.cleanup_thread.start()

    def _get_cache_key(self, lat: float, lon: float) -> str:
        """
        Generate a cache key from lat/lon coordinates.
        Rounds coordinates to reduce cache fragmentation.
        """
        rounded_lat = round(float(lat), self.COORD_PRECISION)
        rounded_lon = round(float(lon), self.COORD_PRECISION)
        return f"{rounded_lat},{rounded_lon}"

    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get current weather data from cache if available and not expired.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with weather data or None if not in cache or expired
        """
        with self.lock:
            key = self._get_cache_key(lat, lon)
            now = time.time()

            # Check if we have current weather data in cache
            if key in self.current_cache:
                cache_entry = self.current_cache[key]
                if now - cache_entry.get("timestamp", 0) < self.CURRENT_WEATHER_TTL:
                    self.cache_hits += 1
                    return cache_entry.get("data")

            # Check if we can extract current weather from forecast data
            if key in self.forecast_cache:
                forecast_entry = self.forecast_cache[key]
                if now - forecast_entry.get("timestamp", 0) < self.FORECAST_WEATHER_TTL:
                    # Get the first hour from the forecast to use as current
                    forecast_data = forecast_entry.get("data", {})
                    if (
                        "forecast" in forecast_data
                        and len(forecast_data["forecast"]) > 0
                    ):
                        # If hours are in forecast, construct current data from first hour
                        first_hour = forecast_data["forecast"][0]
                        current_data = {
                            "coordinates": forecast_data.get("coordinates", {}),
                            "units": forecast_data.get("units", {}),
                            "current": first_hour,
                        }
                        self.cache_hits += 1
                        return current_data

            self.cache_misses += 1
            return None

    def get_forecast_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get forecast weather data from cache if available and not expired.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with weather data or None if not in cache or expired
        """
        with self.lock:
            key = self._get_cache_key(lat, lon)
            now = time.time()

            # Check if we have forecast weather data in cache
            if key in self.forecast_cache:
                cache_entry = self.forecast_cache[key]
                if now - cache_entry.get("timestamp", 0) < self.FORECAST_WEATHER_TTL:
                    self.cache_hits += 1
                    return cache_entry.get("data")

            self.cache_misses += 1
            return None

    def cache_current_weather(self, lat: float, lon: float, data: Dict) -> None:
        """
        Store current weather data in cache.

        Args:
            lat: Latitude
            lon: Longitude
            data: Weather data to cache
        """
        with self.lock:
            key = self._get_cache_key(lat, lon)
            self.current_cache[key] = {"timestamp": time.time(), "data": data}
            self._save_cache()

    def cache_forecast_weather(self, lat: float, lon: float, data: Dict) -> None:
        """
        Store forecast weather data in cache.

        Args:
            lat: Latitude
            lon: Longitude
            data: Weather data to cache
        """
        with self.lock:
            key = self._get_cache_key(lat, lon)
            self.forecast_cache[key] = {"timestamp": time.time(), "data": data}

            # If forecast data contains current hour, also cache as current weather
            if "forecast" in data and len(data["forecast"]) > 0:
                # Use first hour as current weather
                first_hour = data["forecast"][0]
                current_data = {
                    "coordinates": data.get("coordinates", {}),
                    "units": data.get("units", {}),
                    "current": first_hour,
                }
                # Only update if no current data exists or it's older
                if (
                    key not in self.current_cache
                    or time.time() - self.current_cache[key].get("timestamp", 0) > 600
                ):
                    self.current_cache[key] = {
                        "timestamp": time.time(),
                        "data": current_data,
                    }

            self._save_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "current_entries": len(self.current_cache),
                "forecast_entries": len(self.forecast_cache),
                "hit_ratio": (
                    (self.cache_hits / (self.cache_hits + self.cache_misses))
                    if (self.cache_hits + self.cache_misses) > 0
                    else 0
                ),
            }

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        with self.lock:
            now = time.time()
            # Clean current weather cache
            expired_current = [
                key
                for key, entry in self.current_cache.items()
                if now - entry.get("timestamp", 0) >= self.CURRENT_WEATHER_TTL
            ]
            for key in expired_current:
                del self.current_cache[key]

            # Clean forecast weather cache
            expired_forecast = [
                key
                for key, entry in self.forecast_cache.items()
                if now - entry.get("timestamp", 0) >= self.FORECAST_WEATHER_TTL
            ]
            for key in expired_forecast:
                del self.forecast_cache[key]

            if expired_current or expired_forecast:
                self._save_cache()

    def _cleanup_thread(self) -> None:
        """Background thread to periodically clean up expired cache entries."""
        while True:
            time.sleep(self.CLEANUP_INTERVAL)
            try:
                self._cleanup_expired()
            except Exception as e:
                print(f"Error in cache cleanup: {e}")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_data = {
                "current_cache": self.current_cache,
                "forecast_cache": self.forecast_cache,
                "stats": {"hits": self.cache_hits, "misses": self.cache_misses},
            }
            with open(self.CACHE_FILE, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache to disk: {e}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if os.path.exists(self.CACHE_FILE):
                with open(self.CACHE_FILE, "r") as f:
                    cache_data = json.load(f)

                # Restore cache data, filtering out entries that would be expired
                now = time.time()

                current_cache = cache_data.get("current_cache", {})
                self.current_cache = {
                    key: entry
                    for key, entry in current_cache.items()
                    if now - entry.get("timestamp", 0) < self.CURRENT_WEATHER_TTL
                }

                forecast_cache = cache_data.get("forecast_cache", {})
                self.forecast_cache = {
                    key: entry
                    for key, entry in forecast_cache.items()
                    if now - entry.get("timestamp", 0) < self.FORECAST_WEATHER_TTL
                }

                # Restore stats
                stats = cache_data.get("stats", {})
                self.cache_hits = stats.get("hits", 0)
                self.cache_misses = stats.get("misses", 0)

                print(
                    f"Loaded {len(self.current_cache)} current weather and {len(self.forecast_cache)} forecast entries from cache file."
                )
        except Exception as e:
            print(f"Error loading cache from disk: {e}")
            # Initialize empty cache on error
            self.current_cache = {}
            self.forecast_cache = {}


# Create a global instance of the cache
weather_cache = WeatherCache()


def is_near_coordinates(
    lat1: float, lon1: float, lat2: float, lon2: float, threshold_km: float = 5.0
) -> bool:
    """
    Check if two coordinate pairs are within a certain distance of each other.
    Used for more advanced cache matching if exact coordinates don't match.

    Args:
        lat1, lon1: First coordinate pair
        lat2, lon2: Second coordinate pair
        threshold_km: Maximum distance in kilometers for coordinates to be considered near

    Returns:
        True if coordinates are within threshold_km of each other
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance <= threshold_km
