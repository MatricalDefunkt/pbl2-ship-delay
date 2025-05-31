# Port Weather Navigator

Port Weather Navigator is a web application designed to provide insights into port congestion, predict potential shipping delays due to weather, and offer detailed weather forecasts for various ports.

## Features

*   **Port Selection:** Easily select and view data for different ports.
*   **Weather Panel:** Display current and forecasted weather conditions for the selected port.
*   **Delay Prediction:** Predict potential shipping delays based on weather forecasts and historical data.
*   **Congestion Analytics:** Analyze historical and predicted port congestion levels.
*   **Time Series Forecasting:** View time series forecasts for weather and shipping metrics.
*   **Interactive Map:** Visualize port locations and relevant information on a map.

## Screenshots

## Project Structure

The project is divided into two main components:

1.  **Frontend (`port-weather-navigator`):** A React-based single-page application built with Vite and TypeScript, providing the user interface.
2.  **Backend (`pbl2-ship-delay`):** A Python-based server (likely Flask or FastAPI) that serves data, performs predictions, and exposes API endpoints consumed by the frontend.

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
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Setup and run the Backend (`pbl2-ship-delay`):**
    ```bash
    cd pbl2-ship-delay
    pip install -r requirements.txt
    # (Optional) Create and populate .env file if needed for API keys or configurations
    python app.py # Or the main script that starts the backend server
    ```
    The backend server will typically start on a local port (e.g., `http://localhost:5000` or `http://localhost:8000`).

3.  **Setup and run the Frontend (`port-weather-navigator`):**
    Open a new terminal window.
    ```bash
    cd port-weather-navigator
    npm install # or yarn install / pnpm install
    npm run dev
    ```
    The frontend development server will typically start on `http://localhost:5173` (or another port indicated by Vite) and will proxy API requests to the backend.

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
*   **Flask / FastAPI (assumed):** Web framework for building APIs.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Scikit-learn:** For machine learning tasks and preprocessing.
*   **Keras / TensorFlow:** For deep learning models (e.g., RNN, CNN, MLP).
*   **Joblib & Pickle:** For saving and loading machine learning models.

## API Endpoints

The frontend communicates with the backend via a set of API endpoints, likely prefixed with `/api` (as configured in the frontend). While the exact backend implementation details might vary, common endpoints based on the application's features would include:

*   `GET /api/ports`: Fetches a list of available ports.
*   `GET /api/weather/{port_id}?date={date}`: Fetches current and forecasted weather data for a specific port.
*   `POST /api/predict/delay`: Submits data to get a delay prediction.
*   `GET /api/predict/congestion/{port_id}`: Fetches congestion prediction for a specific port.
*   `GET /api/historical/weather/{port_id}`: Fetches historical weather data.
*   `GET /api/historical/shipping/{port_id}`: Fetches historical shipping data.
*   `GET /api/alerts/{port_id}`: Fetches active alerts for a port.
*   `GET /api/forecast/timeseries/{port_id}?metric={metric_name}`: Fetches time series forecast data for a specific metric.

*(Note: These are representative endpoints. Refer to the backend API documentation or source code for the exact specifications.)*

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
