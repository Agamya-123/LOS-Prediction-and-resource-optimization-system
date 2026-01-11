import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Bed, Users, TrendingUp, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Progress } from '@/components/ui/progress';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);

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

  const handleTrainModel = async () => {
    setTraining(true);
    toast.info('Training started... This may take a few moments.');
    try {
      const response = await axios.post(`${API}/train`);
      toast.success(`Model trained! Best model: ${response.data.best_model} with AUC: ${response.data.best_auc.toFixed(4)}`);
      setModelInfo({
        best_model: response.data.best_model,
        best_auc: response.data.best_auc,
        feature_importance: response.data.feature_importance,
        model_comparison: response.data.model_comparison
      });
    } catch (error) {
      toast.error('Training failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setTraining(false);
      fetchData();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-slate-500">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-900 tracking-tight">Dashboard</h1>
          <p className="text-base text-slate-600 mt-1">Real-time hospital bed management and predictions</p>
        </div>
        {!modelInfo && (
          <Button
            onClick={handleTrainModel}
            disabled={training}
            data-testid="train-model-button"
            className="bg-teal-600 hover:bg-teal-700"
          >
            {training ? 'Training...' : 'Train ML Model'}
          </Button>
        )}
      </div>

      {/* Model Status Alert */}
      {!modelInfo && (
        <Card className="border-amber-200 bg-amber-50">
          <CardContent className="p-6 flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-amber-900">Model Not Trained</h3>
              <p className="text-sm text-amber-700 mt-1">
                The ML model needs to be trained before making predictions. Click "Train ML Model" to generate dataset and train the model.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card data-testid="stat-total-beds" className="border-l-4 border-l-teal-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600 flex items-center">
              <Bed className="w-4 h-4 mr-2" />
              Total Beds
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono text-slate-900">{stats?.total_beds || 0}</div>
          </CardContent>
        </Card>

        <Card data-testid="stat-occupied-beds" className="border-l-4 border-l-red-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600 flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              Occupied
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono text-slate-900">{stats?.occupied_beds || 0}</div>
          </CardContent>
        </Card>

        <Card data-testid="stat-available-beds" className="border-l-4 border-l-emerald-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600 flex items-center">
              <Bed className="w-4 h-4 mr-2" />
              Available
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono text-slate-900">{stats?.available_beds || 0}</div>
          </CardContent>
        </Card>

        <Card data-testid="stat-occupancy-rate" className="border-l-4 border-l-sky-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Occupancy Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold font-mono text-slate-900">{stats?.occupancy_rate || 0}%</div>
            <Progress value={stats?.occupancy_rate || 0} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Model Info & Patient Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Performance */}
        {modelInfo && (
          <Card data-testid="model-info-card">
            <CardHeader>
              <CardTitle className="text-xl">Model Performance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                <div>
                  <p className="text-sm text-slate-600">Best Model</p>
                  <p className="text-lg font-bold text-slate-900">{modelInfo.best_model}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-slate-600">ROC AUC</p>
                  <p className="text-lg font-bold font-mono text-emerald-600">{modelInfo.best_auc?.toFixed(4)}</p>
                </div>
              </div>

              {modelInfo.model_comparison && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-slate-700">Model Comparison</p>
                  {Object.entries(modelInfo.model_comparison).map(([name, metrics]) => (
                    <div key={name} className="flex justify-between items-center p-3 bg-slate-50 rounded border">
                      <span className="text-sm font-medium text-slate-700">{name}</span>
                      <span className="text-sm font-mono text-slate-600">AUC: {metrics.auc.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Patient Statistics */}
        <Card data-testid="patient-stats-card">
          <CardHeader>
            <CardTitle className="text-xl flex items-center">
              <Users className="w-5 h-5 mr-2" />
              Patient Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center p-4 bg-slate-50 rounded-lg border">
              <span className="text-sm font-medium text-slate-700">Total Patients</span>
              <span className="text-2xl font-bold font-mono text-slate-900">{stats?.total_patients || 0}</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                <p className="text-xs text-slate-600 mb-1">Short Stay (0-5d)</p>
                <p className="text-2xl font-bold font-mono text-emerald-700">{stats?.short_stay_count || 0}</p>
              </div>
              <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                <p className="text-xs text-slate-600 mb-1">Long Stay (6+d)</p>
                <p className="text-2xl font-bold font-mono text-amber-700">{stats?.long_stay_count || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Feature Importance */}
      {modelInfo?.feature_importance && (
        <Card data-testid="feature-importance-card">
          <CardHeader>
            <CardTitle className="text-xl">Feature Importance</CardTitle>
            <p className="text-sm text-slate-600">Top factors affecting stay duration predictions</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(modelInfo.feature_importance).slice(0, 8).map(([feature, importance], idx) => (
                <div key={feature} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-slate-700">{feature.replace(/_/g, ' ').toUpperCase()}</span>
                    <span className="font-mono text-slate-600">{(importance * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-teal-600 rounded-full"
                      style={{ width: `${importance * 100}%` }}
                    />
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

export default Dashboard;
