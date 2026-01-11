import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from '@/pages/Dashboard';
import BedManagement from '@/pages/BedManagement';
import PredictStay from '@/pages/PredictStay';
import PatientRecords from '@/pages/PatientRecords';
import Analytics from '@/pages/Analytics';
import Layout from '@/components/Layout';
import { Toaster } from '@/components/ui/sonner';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/beds" element={<BedManagement />} />
            <Route path="/predict" element={<PredictStay />} />
            <Route path="/patients" element={<PatientRecords />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </Layout>
      </BrowserRouter>
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
