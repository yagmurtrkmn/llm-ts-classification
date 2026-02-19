# tasks.py

from textwrap import dedent



TASK_CONFIGS = {

    # ==========================================================
    # 1) FIVE-CLASS CLASSIFICATION
    # ==========================================================
    "5_class": {
        "description": "Classify the time series into one of five high-level structural categories.",
        "labels": [
            "Deterministic Trend",
            "Stochastic Trend",
            "Structural Break",
            "Volatility",
            "Anomaly"
        ],
        "hierarchy": dedent("""
        High-Level Time-Series Structure:

        1) Deterministic Trend: These trends follow a predictable mathematical function and do not change randomly over time.
           - Linear Trend: The variable increases or decreases at a constant rate over time.
           - Quadratic Trend: The variable follows a curved trajectory, typically accelerating or decelerating over time.
           - Cubic Trend: It allows for more complex patterns, such as changes in direction, making it useful in cases where the trend is not strictly linear or quadratic.
           - Exponential Trend: The variable grows or declines exponentially, often observed in financial markets or population studies.
           - Damped Trend: The trend starts strong but slows over time, often seen in diffusion processes like technology adoption.

        2) Stochastic Trend: These trends incorporate random fluctuations and are influenced by external shocks, making them unpredictable.

        3) Structural Break: These are abrupt changes in the underlying data-generating process. They can manifest as:
           - Mean Shift: A sudden or gradual change in the average (mean) level of a time series or dataset over time.
           - Variance Shift: The variability of the data increases or decreases at a certain point.
           - Trend Shift: A significant change in the direction, magnitude, or behavior of a trend over time.

        4) Volatility: Refers to the degree of dispersion or fluctuation in a time series over time. It typically exhibits volatility clustering, meaning large changes cluster together and small changes cluster together, and may also display a leverage effect, where adverse (negative) returns have a stronger impact on subsequent volatility than favorable (positive) returns.

        5) Anomaly: A data point or pattern that deviates significantly from the expected behavior in a dataset.
           - Point Anomaly: A single data point that is significantly different from the rest.
           - Collective Anomaly: A group of data points that together form an unusual pattern, even if individual points are not outliers.
           - Contextual Anomaly: A data point that appears abnormal only within a specific context, such as a seasonal pattern.
        """).strip()
    },

    # ==========================================================
    # 2) NINE-CLASS CLASSIFICATION
    # ==========================================================
    "9_class": {
        "description": "Classify the time series into one of nine structural categories.",
        "labels": [
            "Deterministic Trend",
            "Stochastic Trend",
            "Mean Shift",
            "Variance Shift",
            "Trend Shift",
            "Volatility",
            "Point Anomaly",
            "Collective Anomaly",
            "Contextual Anomaly"
        ],
        "hierarchy": dedent("""
        Here is the 9 Structural Categories:

        1) Deterministic Trend: These trends follow a predictable mathematical function and do not change randomly over time. This can be seen in different kinds
           - Linear Trend: The variable increases or decreases at a constant rate over time.
           - Quadratic Trend: The variable follows a curved trajectory, typically accelerating or decelerating over time.
           - Cubic Trend: It allows for more complex patterns, such as changes in direction, making it useful in cases where the trend is not strictly linear or quadratic.
           - Exponential Trend: The variable grows or declines exponentially, often observed in financial markets or population studies.
           - Damped Trend: The trend starts strong but slows over time, often seen in diffusion processes like technology adoption.

        2) Stochastic Trend: These trends incorporate random fluctuations and are influenced by external shocks, making them unpredictable.
                            
        Structural Break: These are abrupt changes in the underlying data-generating process.
        3) Mean Shift: A sudden or gradual change in the average (mean) level of a time series or dataset over time.
        4) Variance Shift: The variability of the data increases or decreases at a certain point.
        5) Trend Shift: A significant change in the direction, magnitude, or behavior of a trend over time.

        6) Volatility: Refers to the degree of dispersion or fluctuation in a time series over time. It typically exhibits volatility clustering, meaning large changes cluster together and small changes cluster together, and may also display a leverage effect, where adverse (negative) returns have a stronger impact on subsequent volatility than favorable (positive) returns.
                            
        Anomaly: A data point or pattern that deviates significantly from the expected behavior in a dataset.
        7) Point Anomaly: A single data point that is significantly different from the rest.
        8) Collective Anomaly: A group of data points that together form an unusual pattern, even if individual points are not outliers.
        9) Contextual Anomaly: A data point that appears abnormal only within a specific context, such as a seasonal pattern.                    
        """).strip()
    },

    # ==========================================================
    # 3) FIVE-CLASS TREND CLASSIFICATION
    # ==========================================================
    "trend_5_class": {
        "description": "Classify the deterministic trend type.",
        "labels": [
            "Linear Trend",
            "Quadratic Trend",
            "Cubic Trend",
            "Exponential Trend",
            "Damped Trend"
        ],
        "hierarchy": dedent("""
        Deterministic Trend : These trends follow a predictable mathematical function and do not change randomly over time. This can be seen in different kinds:

        1) Linear Trend: The variable increases or decreases at a constant rate over time.
        2) Quadratic Trend: The variable follows a curved trajectory, typically accelerating or decelerating over time.
        3) Cubic Trend: It allows for more complex patterns, such as changes in direction, making it useful in cases where the trend is not strictly linear or quadratic.
        4) Exponential Trend: The variable grows or declines exponentially, often observed in financial markets or population studies.
        5) Damped Trend: The trend starts strong but slows over time, often seen in diffusion processes like technology adoption.
        """).strip()
    },

    # ==========================================================
    # 4) THREE-CLASS STRUCTURAL BREAK CLASSIFICATION
    # ==========================================================
    "break_3_class": {
        "description": "Classify the structural break type.",
        "labels": [
            "Mean Shift",
            "Variance Shift",
            "Trend Shift"
        ],
        "hierarchy": dedent("""
        Structural Break: These are abrupt changes in the underlying data-generating process. This can be seen in different kinds:
                            
        1) Mean Shift: A sudden or gradual change in the average (mean) level of a time series or dataset over time.
        2) Variance Shift: The variability of the data increases or decreases at a certain point.
        3) Trend Shift: A significant change in the direction, magnitude, or behavior of a trend over time.
        """).strip()
    },

    # ==========================================================
    # 5) THREE-CLASS ANOMALY CLASSIFICATION
    # ==========================================================
    "anomaly_3_class": {
        "description": "Classify the anomaly type.",
        "labels": [
            "Point Anomaly",
            "Collective Anomaly",
            "Contextual Anomaly"
        ],
        "hierarchy": dedent("""
        Anomaly: A data point or pattern that deviates significantly from the expected behavior in a dataset. This can be seen in different kinds:
                            
        1) Point Anomaly: A single data point that is significantly different from the rest.
        2) Collective Anomaly: A group of data points that together form an unusual pattern, even if individual points are not outliers.
        3) Contextual Anomaly: A data point that appears abnormal only within a specific context, such as a seasonal pattern.    
        """).strip()
    },

    # ==========================================================
    # 6) TWO-CLASS STATIONARITY CLASSIFICATION
    # ==========================================================
    "2_class_stationarity": {
        "description": "Classify whether the time series is stationary or non-stationary.",
        "labels": [
            "Stationary",
            "Non-Stationary"
        ],
        "hierarchy": dedent("""
                            
        1) Stationary : A stochastic process whose statistical properties, such as mean and variance, do not change over time.
        2) Non-Stationary : A stochastic process whose statistical properties, such as mean and variance, change over time.     
        """).strip()
    }

}
