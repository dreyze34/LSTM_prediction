# Performance metrics in Machine Learning

This project is a Python script that takes a time series of a house's energy consumption as input, train a LSTM model on the data and return a one-day prediction of the next consumption.

## Prerequisites

Make sure you have the following installed before proceeding:

- Python 3.x
- `pip` (the Python package installer)

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create a virtual environment for the project:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

4. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the `main.py` script, use the following command with the virtual environment activated:

```bash
python src/main.py
