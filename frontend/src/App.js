import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from '@/pages/Dashboard';
import BedManagement from '@/pages/BedManagement';
import PredictStay from '@/pages/PredictStay';
import PatientRecords from '@/pages/PatientRecords';
import Analytics from '@/pages/Analytics';
import Login from '@/pages/Login';
import Layout from '@/components/Layout';
import { Toaster } from '@/components/ui/sonner';

function App() {
  const [user, setUser] = useState(null);

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  // If not logged in, show Login page
  if (!user) {
    return (
      <div className="App">
        <Login onLogin={handleLogin} />
        <Toaster position="top-right" />
      </div>
    );
  }

  return (
    <div className="App">
      <BrowserRouter>
        <Layout user={user} onLogout={handleLogout}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            {user.permissions.includes('Bed Management') && (
              <Route path="/beds" element={<BedManagement />} />
            )}
            {user.permissions.includes('Predict Stay') && (
              <Route path="/predict" element={<PredictStay />} />
            )}
            <Route path="/patients" element={<PatientRecords />} />
            {user.permissions.includes('Analytics') && (
              <Route path="/analytics" element={<Analytics />} />
            )}
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </Layout>
      </BrowserRouter>
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
