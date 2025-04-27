# time_series.py
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For ARIMA models - will need statsmodels library
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    STATSMODELS_AVAILABLE = True
except ImportError:
    print(
        "Warning: statsmodels not installed. ARIMA forecasting will not be available."
    )
    print("To install: pip install statsmodels>=0.14.0")
    STATSMODELS_AVAILABLE = False


class ARIMAForecaster:
    """
    Time series forecasting using ARIMA (AutoRegressive Integrated Moving Average) models.
    Provides methods to train, save, load, and make predictions with ARIMA models.
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the ARIMA forecaster.

        Args:
            model_dir (str): Directory to save/load trained models
        """
        self.model_dir = model_dir
        self.model = None
        self.order = None
        self.diff_order = 0
        self.model_path = os.path.join(model_dir, "arima_model.pkl")
        self.seasonal_model_path = os.path.join(model_dir, "sarima_model.pkl")

        # Ensure the model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Check if statsmodels is available
        if not STATSMODELS_AVAILABLE:
            print("ARIMA forecasting requires statsmodels library")

    def check_stationarity(self, series):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.

        Args:
            series (pd.Series): Time series data

        Returns:
            tuple: (is_stationary, p_value, test_statistic, critical_values)
        """
        if not STATSMODELS_AVAILABLE:
            print("Cannot check stationarity: statsmodels not available")
            return None

        result = adfuller(series.dropna())
        is_stationary = result[1] < 0.05  # p-value < 0.05 means stationary

        return (is_stationary, result[1], result[0], result[4])

    def train(self, time_series, order=(1, 1, 1), train_test_split=0.8, verbose=True):
        """
        Train an ARIMA model on the provided time series data.

        Args:
            time_series (pd.Series): Time series data with DatetimeIndex
            order (tuple): ARIMA order (p,d,q)
            train_test_split (float): Proportion of data to use for training
            verbose (bool): Whether to print training details

        Returns:
            dict: Training results with metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")

        # Store the order
        self.order = order
        p, d, q = order
        self.diff_order = d

        # Split data into training and testing
        train_size = int(len(time_series) * train_test_split)
        train_data = time_series[:train_size]
        test_data = time_series[train_size:]

        if verbose:
            print(f"Training ARIMA({p},{d},{q})")
            print(
                f"Training data size: {len(train_data)}, Test data size: {len(test_data)}"
            )

        # Fit the model
        try:
            self.model = ARIMA(train_data, order=order)
            self.model_fit = self.model.fit()

            if verbose:
                print(self.model_fit.summary())

            # Make forecasts
            forecast_steps = len(test_data)
            forecast = self.model_fit.forecast(steps=forecast_steps)

            # Ensure forecasts are non-negative
            forecast = np.maximum(0, forecast)

            # Evaluate forecast
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data, forecast)

            # Save the trained model
            self.save_model()

            # Return evaluation results
            return {
                "order": order,
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "train_size": len(train_data),
                "test_size": len(test_data),
            }

        except Exception as e:
            if verbose:
                print(f"Error training ARIMA model: {e}")
            return {"error": str(e)}

    def train_seasonal(
        self,
        time_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        train_test_split=0.8,
        verbose=True,
    ):
        """
        Train a Seasonal ARIMA model (SARIMA).

        Args:
            time_series (pd.Series): Time series data with DatetimeIndex
            order (tuple): ARIMA order (p,d,q)
            seasonal_order (tuple): Seasonal order (P,D,Q,s)
            train_test_split (float): Proportion of data to use for training
            verbose (bool): Whether to print training details

        Returns:
            dict: Training results with metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMA forecasting")

        # Store orders
        self.order = order
        self.seasonal_order = seasonal_order
        p, d, q = order
        P, D, Q, s = seasonal_order
        self.diff_order = d

        # Split data
        train_size = int(len(time_series) * train_test_split)
        train_data = time_series[:train_size]
        test_data = time_series[train_size:]

        if verbose:
            print(f"Training SARIMA({p},{d},{q})x({P},{D},{Q}){s}")
            print(
                f"Training data size: {len(train_data)}, Test data size: {len(test_data)}"
            )

        # Fit the model
        try:
            self.model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            self.model_fit = self.model.fit(disp=False)

            if verbose:
                print(self.model_fit.summary())

            # Make forecasts
            forecast_steps = len(test_data)
            forecast = self.model_fit.forecast(steps=forecast_steps)

            # Ensure forecasts are non-negative
            forecast = np.maximum(0, forecast)

            # Evaluate forecast
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data, forecast)

            # Save the seasonal model
            self.save_model(seasonal=True)

            # Return evaluation results
            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "train_size": len(train_data),
                "test_size": len(test_data),
            }

        except Exception as e:
            if verbose:
                print(f"Error training SARIMA model: {e}")
            return {"error": str(e)}

    def save_model(self, seasonal=False):
        """
        Save the trained model.

        Args:
            seasonal (bool): Whether to save as seasonal model
        """
        if self.model_fit is None:
            print("No trained model to save")
            return

        path = self.seasonal_model_path if seasonal else self.model_path
        try:
            self.model_fit.save(path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, seasonal=False):
        """
        Load a trained model.

        Args:
            seasonal (bool): Whether to load seasonal model

        Returns:
            bool: True if loading successful, False otherwise
        """
        if not STATSMODELS_AVAILABLE:
            print("Cannot load model: statsmodels not available")
            return False

        path = self.seasonal_model_path if seasonal else self.model_path

        try:
            if os.path.exists(path):
                if seasonal:
                    self.model_fit = sm.load(path)
                    print(f"Loaded seasonal ARIMA model from {path}")
                else:
                    self.model_fit = sm.load(path)
                    print(f"Loaded ARIMA model from {path}")
                return True
            else:
                print(f"Model file not found: {path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def forecast(self, steps, seasonal=False, include_confidence=True, alpha=0.05):
        """
        Generate forecasts for future periods.

        Args:
            steps (int): Number of steps to forecast
            seasonal (bool): Whether to use seasonal model
            include_confidence (bool): Whether to include confidence intervals
            alpha (float): Significance level for confidence intervals (0.05 = 95% CI)

        Returns:
            dict: Forecast results with predictions and optionally confidence intervals
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels is required for forecasting"}

        # Load the appropriate model if not available
        if self.model_fit is None:
            success = self.load_model(seasonal=seasonal)
            if not success:
                return {"error": "No trained model available"}

        try:
            # Generate forecast
            forecast_result = self.model_fit.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean

            # Ensure non-negative predictions
            forecast = np.maximum(0, forecast)

            result = {"forecast": forecast.tolist()}

            # Include confidence intervals if requested
            if include_confidence:
                conf_int = forecast_result.conf_int(alpha=alpha)

                # Ensure lower bound is non-negative
                conf_int.iloc[:, 0] = np.maximum(0, conf_int.iloc[:, 0])

                result["lower_bound"] = conf_int.iloc[:, 0].tolist()
                result["upper_bound"] = conf_int.iloc[:, 1].tolist()
                result["confidence_level"] = 1 - alpha

            return result

        except Exception as e:
            return {"error": str(e)}

    def plot_forecast(
        self,
        historical_data,
        forecast_steps,
        seasonal=False,
        output_file=None,
        show_plot=True,
        figure_size=(12, 6),
    ):
        """
        Plot historical data and future forecast.

        Args:
            historical_data (pd.Series): Historical time series data
            forecast_steps (int): Number of steps to forecast
            seasonal (bool): Whether to use seasonal model
            output_file (str): Path to save the plot image (optional)
            show_plot (bool): Whether to display the plot
            figure_size (tuple): Figure size in inches

        Returns:
            dict: Forecast results including plot path if saved
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels is required for forecasting"}

        # Generate the forecast
        forecast_result = self.forecast(forecast_steps, seasonal=seasonal)

        if "error" in forecast_result:
            return forecast_result

        # Create forecast dates
        last_date = historical_data.index[-1]

        # Determine the frequency of the historical data
        if isinstance(historical_data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(historical_data.index)
            if freq is None:
                # Try to guess the frequency from the last few observations
                freq = pd.infer_freq(historical_data.index[-5:])

            if freq is None:
                # Default to daily if can't determine
                freq = "D"
                print("Warning: Could not determine frequency, using daily (D)")

            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq=freq,
            )
        else:
            # For non-datetime index, use numeric index
            forecast_dates = range(
                len(historical_data), len(historical_data) + forecast_steps
            )

        # Create the forecast series
        forecast_series = pd.Series(forecast_result["forecast"], index=forecast_dates)

        # Create confidence interval series if available
        if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
            lower_series = pd.Series(
                forecast_result["lower_bound"], index=forecast_dates
            )
            upper_series = pd.Series(
                forecast_result["upper_bound"], index=forecast_dates
            )

        # Plot the results
        plt.figure(figsize=figure_size)
        plt.plot(historical_data, label="Historical Data")
        plt.plot(forecast_series, color="red", label="Forecast")

        if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
            plt.fill_between(
                forecast_dates,
                lower_series,
                upper_series,
                color="pink",
                alpha=0.3,
                label=f'{int((1-forecast_result["confidence_level"])*100)}% Confidence Interval',
            )

        # Add labels and title
        plt.xlabel("Date")
        plt.ylabel("Value")
        if seasonal:
            title = f"SARIMA Forecast ({forecast_steps} steps)"
        else:
            p, d, q = self.order if self.order else (0, 0, 0)
            title = f"ARIMA({p},{d},{q}) Forecast ({forecast_steps} steps)"

        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot if requested
        if output_file:
            plt.savefig(output_file)
            forecast_result["plot_path"] = output_file

        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

        return forecast_result


def prepare_daily_delay_time_series(
    ops_df,
    target_column="total_weather_delay_hrs",
    timestamp_column="arrival_timestamp",
    base_datetime=None,
    base_time_column="arrival_time",
):
    """
    Prepares daily aggregated time series from operations data for ARIMA modeling.

    Args:
        ops_df (pd.DataFrame): Operations data
        target_column (str): Column containing delay values
        timestamp_column (str): Column containing timestamps (if already exists)
        base_datetime (datetime): Base datetime to convert hour values (if needed)
        base_time_column (str): Column containing hour values (if timestamps need to be created)

    Returns:
        pd.Series: Daily aggregated time series of the target column
    """
    # Ensure we have a timestamp column
    if timestamp_column not in ops_df.columns:
        if base_datetime is None:
            base_datetime = datetime(2023, 1, 1)  # Default if not provided
            print(f"Using default base datetime: {base_datetime}")

        if base_time_column not in ops_df.columns:
            raise ValueError(
                f"Neither {timestamp_column} nor {base_time_column} found in DataFrame"
            )

        # Convert hours to timestamps
        ops_df[timestamp_column] = ops_df[base_time_column].apply(
            lambda h: base_datetime + timedelta(hours=h)
        )

    # Set the timestamp as index
    ops_time_series = ops_df.set_index(timestamp_column)

    # Create daily aggregated series (using mean as the aggregation method)
    daily_delays = ops_time_series[target_column].resample("D").mean().fillna(0)

    print(f"Created daily time series with {len(daily_delays)} data points")
    return daily_delays


def create_arima_endpoint(app, models_dict):
    """
    Add ARIMA forecasting endpoints to a Flask app.

    Args:
        app: Flask application
        models_dict: Dictionary to store loaded models
    """
    from flask import request, jsonify
    import time

    @app.route("/forecast/arima", methods=["POST"])
    def arima_forecast():
        """Endpoint to make time series forecasts using ARIMA."""
        if not STATSMODELS_AVAILABLE:
            return jsonify({"error": "statsmodels not installed on the server"}), 500

        start_time = time.time()

        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        # Validate request
        required_fields = ["forecast_days", "use_seasonal"]
        if not all(field in data for field in required_fields):
            return (
                jsonify(
                    {"error": f"Missing required fields. Required: {required_fields}"}
                ),
                400,
            )

        forecast_days = int(data["forecast_days"])
        use_seasonal = data.get("use_seasonal", False)

        # Initialize forecaster
        if "arima_forecaster" not in models_dict:
            models_dict["arima_forecaster"] = ARIMAForecaster()

        forecaster = models_dict["arima_forecaster"]

        try:
            # Check if we need to train a model if none is loaded
            if forecaster.model_fit is None:
                # Try to load existing model
                success = forecaster.load_model(seasonal=use_seasonal)

                # If no model available, train on synthetic data
                if not success:
                    print("No pre-trained model available. Training a new model...")

                    # Load synthetic data for training
                    try:
                        synthetic_data_path = "synthetic_operations_log.csv"
                        if os.path.exists(synthetic_data_path):
                            ops_df = pd.read_csv(synthetic_data_path)

                            # Convert data to time series
                            daily_delays = prepare_daily_delay_time_series(
                                ops_df,
                                target_column="total_weather_delay_hrs",
                                timestamp_column=(
                                    "arrival_timestamp"
                                    if "arrival_timestamp" in ops_df.columns
                                    else None
                                ),
                            )

                            # Train the model
                            if use_seasonal:
                                forecaster.train_seasonal(
                                    daily_delays,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 7),
                                )
                            else:
                                forecaster.train(daily_delays, order=(1, 1, 1))
                        else:
                            # If no data available, create synthetic time series
                            days = 180
                            dates = pd.date_range(
                                start="2023-01-01", periods=days, freq="D"
                            )
                            y = (
                                np.sin(np.linspace(0, 15, days)) * 5
                                + np.random.normal(0, 1, days)
                                + 5
                            )
                            y = y + np.linspace(0, 3, days)  # Add trend
                            y = np.maximum(0, y)  # Ensure non-negative

                            # Create time series
                            time_series = pd.Series(y, index=dates)

                            # Train the model
                            if use_seasonal:
                                forecaster.train_seasonal(
                                    time_series,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 7),
                                )
                            else:
                                forecaster.train(time_series, order=(1, 1, 1))
                    except Exception as e:
                        print(f"Error training model: {e}")
                        return (
                            jsonify({"error": f"Failed to train model: {str(e)}"}),
                            500,
                        )

            # Generate forecast
            forecast_result = forecaster.forecast(
                steps=forecast_days, seasonal=use_seasonal
            )

            if "error" in forecast_result:
                return jsonify({"error": forecast_result["error"]}), 500

            # Prepare response
            last_date = datetime.now().date()
            forecast_dates = [
                (last_date + timedelta(days=i + 1)).isoformat()
                for i in range(forecast_days)
            ]

            response = {
                "model_used": "SARIMA" if use_seasonal else "ARIMA",
                "forecast_days": forecast_days,
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_result["forecast"],
                "processing_time_seconds": round(time.time() - start_time, 4),
            }

            # Include confidence intervals if available
            if "lower_bound" in forecast_result and "upper_bound" in forecast_result:
                response["confidence_intervals"] = {
                    "lower_bound": forecast_result["lower_bound"],
                    "upper_bound": forecast_result["upper_bound"],
                    "confidence_level": forecast_result["confidence_level"],
                }

            return jsonify(response), 200

        except Exception as e:
            print(f"Error in forecast endpoint: {e}")
            import traceback

            traceback.print_exc()
            return (
                jsonify({"error": f"An error occurred during forecasting: {str(e)}"}),
                500,
            )

    # Don't override the original health check, just add ARIMA info to models_dict
    # Instead of redefining the route, let's just update the models dictionary
    models_dict["time_series_forecasting"] = {
        "available": STATSMODELS_AVAILABLE,
        "methods": ["ARIMA", "SARIMA"] if STATSMODELS_AVAILABLE else [],
        "endpoint": "/forecast/arima",
    }


if __name__ == "__main__":
    # Test code to run when this module is executed directly
    import pandas as pd
    import numpy as np

    # Generate sample time series data (sine wave with noise)
    days = 200
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    y = np.sin(np.linspace(0, 15, days)) * 5 + np.random.normal(0, 1, days) + 5
    # Add trend
    y = y + np.linspace(0, 3, days)
    y = np.maximum(0, y)  # Ensure non-negative

    # Create time series
    time_series = pd.Series(y, index=dates)
    print(f"Created sample time series with {len(time_series)} data points")

    # Initialize and train ARIMA forecaster
    forecaster = ARIMAForecaster()

    # Check stationarity
    stationarity_result = forecaster.check_stationarity(time_series)
    if stationarity_result:
        is_stationary, p_value, test_statistic, critical_values = stationarity_result
        print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
        print(f"p-value: {p_value:.4f}")

    # Train ARIMA model
    train_result = forecaster.train(time_series, order=(2, 1, 2))
    print("Training results:", train_result)

    # Make and plot forecast
    forecast_steps = 30
    forecast_result = forecaster.plot_forecast(
        time_series, forecast_steps, output_file="arima_forecast.png"
    )

    print("Forecast:", forecast_result["forecast"][:5], "...")
    if "plot_path" in forecast_result:
        print(f"Forecast plot saved to {forecast_result['plot_path']}")
