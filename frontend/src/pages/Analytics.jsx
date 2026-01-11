import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Analytics = () => {
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [statsRes, modelRes] = await Promise.all([
        axios.get(`${API}/analytics/stats`),
        axios.get(`${API}/model/info`).catch(() => ({ data: null }))
      ]);
      setStats(statsRes.data);
      setModelInfo(modelRes.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading analytics...</div>;
  }

  const bedStatusData = [
    { name: 'Available', value: stats?.available_beds || 0, color: '#10B981' },
    { name: 'Occupied', value: stats?.occupied_beds || 0, color: '#EF4444' },
    { name: 'Cleaning', value: stats?.cleaning_beds || 0, color: '#F59E0B' }
  ];

  const predictionData = [
    { name: 'Short Stay', count: stats?.short_stay_count || 0, color: '#10B981' },
    { name: 'Long Stay', count: stats?.long_stay_count || 0, color: '#F59E0B' }
  ];

  const modelComparisonData = modelInfo?.model_comparison
    ? Object.entries(modelInfo.model_comparison).map(([name, metrics]) => ({
        name,
        auc: metrics.auc,
        accuracy: metrics.accuracy
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-900 tracking-tight">Analytics</h1>
        <p className="text-base text-slate-600 mt-1">Hospital performance and ML model insights</p>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Bed Status Distribution */}
        <Card data-testid="bed-status-chart">
          <CardHeader>
            <CardTitle>Bed Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={bedStatusData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {bedStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Patient Stay Predictions */}
        <Card data-testid="predictions-chart">
          <CardHeader>
            <CardTitle>Patient Stay Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" stroke="#64748B" />
                <YAxis stroke="#64748B" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFF',
                    border: '1px solid #E2E8F0',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {predictionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Model Comparison */}
        {modelInfo && modelComparisonData.length > 0 && (
          <Card data-testid="model-comparison-chart" className="md:col-span-2">
            <CardHeader>
              <CardTitle>ML Model Performance Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="name" stroke="#64748B" />
                  <YAxis stroke="#64748B" domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFF',
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="auc" fill="#0D9488" name="ROC AUC" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="accuracy" fill="#0EA5E9" name="Accuracy" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Feature Importance */}
      {modelInfo?.feature_importance && (
        <Card data-testid="feature-importance-analytics">
          <CardHeader>
            <CardTitle>Top Feature Importance</CardTitle>
            <p className="text-sm text-slate-600">Factors that most influence length of stay predictions</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(modelInfo.feature_importance)
                .slice(0, 10)
                .map(([feature, importance], idx) => (
                  <div key={feature} className="flex items-center space-x-4">
                    <div className="w-8 text-center">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-teal-100 text-teal-700 text-xs font-bold">
                        {idx + 1}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-slate-700">
                          {feature.replace(/_/g, ' ').toUpperCase()}
                        </span>
                        <span className="text-sm font-mono text-slate-600">
                          {(importance * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-teal-600 to-teal-400 rounded-full"
                          style={{ width: `${importance * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Analytics;
