# Port Weather Navigator

Port Weather Navigator is a web application designed to provide insights into port congestion, predict potential shipping delays due to weather, and offer detailed weather forecasts for various ports.

## Features

*   **Port Selection:** Easily select and view data for different ports.
*   **Weather Panel:** Display current and forecasted weather conditions for the selected port.
*   **Delay Prediction:** Predict potential shipping delays based on weather forecasts and historical data.
*   **Congestion Analytics:** Analyze historical and predicted port congestion levels.
*   **Time Series Forecasting:** View time series forecasts for weather and shipping metrics.

## Screenshots

![Dashboard](https://github.com/user-attachments/assets/0883d939-5786-48e1-b093-13955a2963b5)
![Predictions Dashboard](https://github.com/user-attachments/assets/de6ca823-d992-4032-af63-8afc22ecbae2)
![Model List](https://github.com/user-attachments/assets/18590fa8-922e-4daa-8726-adb88ba35d32)
![Using Another Port](https://github.com/user-attachments/assets/4bfb0c34-fe84-4efb-ac3e-fcb58a33573c)
![Predictions Page Using Another Port](https://github.com/user-attachments/assets/5871842f-e2e3-44c5-95e5-4f57af6b0be7)


## Project Structure

The project is divided into two main components:

1.  **[Frontend](https://github.com/MatricalDefunkt/port-weather-navigator) (`port-weather-navigator`):** A React-based single-page application built with Vite and TypeScript, providing the user interface.
2.  **[Backend](https://github.com/MatricalDefunkt/pbl2-ship-delay) (`pbl2-ship-delay`):** A Python-based server that serves data, performs predictions, and exposes API endpoints consumed by the frontend.

## Prerequisites

### Frontend:
*   Node.js (v18 or later recommended)
*   npm (or yarn/pnpm)

### Backend:
*   Python (3.10 or later recommended)
*   pip

## Cloning and Running the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MatricalDefunkt/port-weather-navigator
    git clone https://github.com/MatricalDefunkt/pbl2-ship-delay
    ```

2.  **Setup and run the Backend (`pbl2-ship-delay`):**
    ```bash
    cd pbl2-ship-delay
    pip install -r requirements.txt
    python app.py
    ```
    The backend server will typically start on a local port `http://localhost:5000`.

3.  **Setup and run the Frontend (`port-weather-navigator`):**
    Open a new terminal window.
    ```bash
    cd port-weather-navigator
    npm install # or yarn install / pnpm install
    npm run dev
    ```
    The frontend development server will typically start on `http://localhost:5173` and will proxy API requests to the backend.

## Technologies Used

### Frontend (`port-weather-navigator`):
*   **React:** A JavaScript library for building user interfaces.
*   **Vite:** A fast build tool and development server for modern web projects.
*   **TypeScript:** A typed superset of JavaScript.
*   **Tailwind CSS:** A utility-first CSS framework.
*   **Shadcn/ui:** Re-usable components built using Radix UI and Tailwind CSS.
*   **Recharts:** A composable charting library.
*   **TanStack Query:** For data fetching, caching, and state management.
*   **React Router DOM:** For client-side routing.

### Backend (`pbl2-ship-delay`):
*   **Python:** Main programming language.
*   **Flask:** Web framework for building APIs.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Scikit-learn:** For machine learning tasks and preprocessing.
*   **Keras / TensorFlow:** For deep learning models (e.g., RNN, CNN, MLP).
*   **Joblib & Pickle:** For saving and loading machine learning models.

## API Endpoints

The frontend communicates with the backend via a set of API endpoints. The main Flask application (`app.py`) in the `pbl2-ship-delay` directory defines these routes.

*   `GET /`: Health check for the prediction server.
    *   Returns: JSON with status, message, and list of available models.
*   `POST /predict`: Endpoint to make delay predictions.
    *   Expects: JSON payload with `vessel_type`, `teu`, `arrival_timestamp_str`, `hourly_weather_forecast`, and `model_name`.
    *   Returns: JSON with prediction, model used, input summary, and processing time.
*   `GET /weather`: Fetches weather data from Open-Meteo API (with caching).
    *   Query Parameters: `lat` (latitude), `lon` (longitude), `forecast` (boolean, default true).
    *   Returns: JSON with weather data (current or forecast).
*   `GET /weather/cache/stats`: Retrieves statistics about the weather cache.
    *   Returns: JSON with cache stats and configuration.
*   `POST /forecast/arima/train`: Trains a new ARIMA model.
    *   Expects: JSON payload with `time_series_data` (list of values), `order` (e.g., [1,1,1]), `train_test_split`.
    *   Returns: JSON with training results (RMSE, MAE, etc.) or error.
*   `POST /forecast/sarima/train`: Trains a new SARIMA (Seasonal ARIMA) model.
    *   Expects: JSON payload with `time_series_data`, `order`, `seasonal_order` (e.g., [1,1,1,7]), `train_test_split`.
    *   Returns: JSON with training results or error.
*   `POST /forecast/arima/predict`: Makes predictions using a trained ARIMA model.
    *   Expects: JSON payload with `steps` (number of future steps to predict).
    *   Returns: JSON with forecasted values or error.
*   `POST /forecast/sarima/predict`: Makes predictions using a trained SARIMA model.
    *   Expects: JSON payload with `steps`.
    *   Returns: JSON with forecasted values or error.
*   `GET /forecast/arima/models`: Lists available (trained) ARIMA/SARIMA models.
    *   Returns: JSON with a list of model names and their status.
