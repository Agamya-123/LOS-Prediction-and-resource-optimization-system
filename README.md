# Hospital Bed Management & LOS Prediction System

An AI-powered application designed to optimize hospital bed allocation and predict patient Length of Stay (LOS) to improve resource utilization.

## 🚀 Features

*   **Real-time Bed Management**: Track bed status (Available, Occupied, Cleaning) in real-time.
*   **AI-Driven LOS Prediction**: Predict patient Length of Stay using Machine Learning models (Voting Ensemble: XGBoost, Random Forest, HistGradientBoosting) based on patient data.
*   **Clinical Anomaly Detection**: Unsupervised Isolation Forest acts as a safety net to flag multivariate clinical outliers and data entry errors.
*   **Actionable AI Recommendations**: Active Clinical Decision Support System (CDSS) evaluates data to generate real-time medical directives.
*   **Personalized eXplainable AI (XAI)**: Utilizes Game Theory mathematics via SHAP to deconstruct neural weights and provide complete transparency into individual patient risk drivers.
*   **Interactive Dashboard**: Visual analytics for bed occupancy, patient demographics, and prediction statistics.
*   **Patient Management**: Streamlined admission and discharge workflows.
*   **Smart Allocation**: Intelligent bed assignment suggestions based on predicted stay duration.

## 🛠️ Tech Stack

### Backend
*   **Framework**: FastAPI (Python)
*   **Database**: MongoDB (Motor for async driver)
*   **ML/Data**: Scikit-learn, Pandas, NumPy, XGBoost
*   **Authentication**: JWT (JSON Web Tokens)

### Frontend
*   **Framework**: React.js
*   **Styling**: Tailwind CSS, PostCSS
*   **UI Components**: Radix UI, Lucide React
*   **Visualization**: Recharts
*   **State Management**: React Hooks

## 📦 Installation & Setup

### Prerequisites
*   Python 3.8+
*   Node.js & npm/yarn
*   MongoDB installed and running locally (or a cloud Atlas URI)

### Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure Environment Variables:
    *   Create a `.env` file in the `backend` directory.
    *   Add your MongoDB connection string:
        ```env
        MONGO_URL=mongodb://localhost:27017
        DB_NAME=hospital_db
        ```
5.  Start the server:
    ```bash
    uvicorn server:app --reload
    ```
    The API will run at `http://localhost:8000`.

### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    # or
    yarn install
    ```
3.  Start the development server:
    ```bash
    npm start
    # or
    yarn start
    ```
    The application will open at `http://localhost:3000`.

## 🧠 Machine Learning Model

The system uses a parallel inference ML engine to predict the Length of Stay (Short Stay vs. Long Stay) and provide clinical insights.
*   **LOS Prediction**: Voting Ensemble Classifier combining Random Forest, XGBoost, and HistGradientBoosting.
*   **Anomaly Detection**: Isolation Forest algorithm to flag top 5% of multidimensional clinical outliers.
*   **XAI Explainer**: SHAP TreeExplainer calculates the exact algorithmic marginal contribution of variables for real-time transparency.
*   **Training**: The model can be retrained via the API endpoint `/api/train`.
*   **Input Features**: Age, Gender, Admission Type, Department, Comorbidity, Number of Procedures.

## 🤝 Contributing

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.
