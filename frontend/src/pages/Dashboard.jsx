import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Bed, Users, TrendingUp, AlertCircle, Zap, ShieldCheck, Microscope, Database } from 'lucide-react';
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
    toast.info('AI Optimization engine starting...');
    try {
      const response = await axios.post(`${API}/train`);
      toast.success(`Pipeline optimized! Model: ${response.data.best_model}`);
      setModelInfo({
        best_model: response.data.best_model,
        best_auc: response.data.best_auc,
        feature_importance: response.data.feature_importance,
        model_comparison: response.data.model_comparison
      });
    } catch (error) {
      toast.error('Optimization failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setTraining(false);
      fetchData();
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <Activity className="w-12 h-12 text-teal-500 animate-spin mb-4" />
        <p className="font-bold text-slate-400 uppercase tracking-widest text-xs">Initializing Healthcare OS...</p>
      </div>
    );
  }

  return (
    <div className="max-w-[1600px] mx-auto space-y-6 animate-in fade-in duration-700">
      
      {/* ── Mission Control Header ── */}
      <div className="relative overflow-hidden bg-slate-900 rounded-[2.5rem] p-8 md:p-10 border border-slate-800 shadow-2xl">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-teal-500/10 to-transparent pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="space-y-3">
             <div className="flex items-center gap-2">
                <span className="px-3 py-1 bg-teal-500/20 text-teal-400 text-[10px] font-black tracking-widest uppercase rounded-full border border-teal-500/30">
                  Critical Systems Online
                </span>
             </div>
             <h1 className="text-4xl md:text-5xl font-black text-white tracking-tight">
               Global Hospital <span className="text-teal-400">Overview</span>
             </h1>
             <p className="text-slate-400 max-w-xl text-sm font-medium leading-relaxed">
               Command center for AI-driven patient flow, resource allocation, and clinical model performance monitoring. 
             </p>
          </div>

          <div className="flex flex-col items-end gap-3 w-full md:w-auto">
             <Button
                onClick={handleTrainModel}
                disabled={training}
                className="h-14 px-10 rounded-2xl bg-teal-500 hover:bg-teal-400 text-white font-black shadow-xl shadow-teal-500/20 transition-all active:scale-95 text-lg"
             >
                {training ? (
                  <>
                    <Zap className="w-5 h-5 mr-3 animate-pulse" />
                     OPTIMIZING...
                  </>
                ) : (
                  <>
                    <Microscope className="w-5 h-5 mr-3" />
                    OPTIMIZE AI ENGINE
                  </>
                )}
             </Button>
             <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mr-2">Last Sync: Real-time</p>
          </div>
        </div>
      </div>

      {/* ── Core KPI Hub ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { icon: Bed, label: 'Total Capacity', val: stats?.total_beds, color: 'teal', trend: 'Global' },
          { icon: Activity, label: 'Active Occupancy', val: stats?.occupied_beds, color: 'rose', trend: 'Live' },
          { icon: ShieldCheck, label: 'Patient Headcount', val: stats?.total_patients, color: 'indigo', trend: 'Records' },
          { icon: TrendingUp, label: 'Occupancy Flow', val: `${stats?.occupancy_rate}%`, color: 'amber', trend: 'Rate' },
        ].map((card, i) => (
          <Card key={i} className="border-none shadow-xl rounded-[2rem] overflow-hidden group hover:scale-[1.02] transition-transform duration-500 ring-1 ring-slate-100">
             <CardContent className="p-8">
                <div className="flex justify-between items-start mb-6">
                   <div className={`p-4 rounded-3xl bg-${card.color}-50 text-${card.color}-600 group-hover:scale-110 transition-transform duration-500`}>
                      <card.icon className="w-7 h-7" />
                   </div>
                   <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">{card.trend}</span>
                </div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">{card.label}</p>
                <div className="text-4xl font-black text-slate-800 tracking-tighter">
                   {card.val || '0'}
                </div>
             </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Deep Insights Grid ── */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        
        {/* ML Performance Dashboard (Left) */}
        <div className="xl:col-span-8 space-y-8">
           {!modelInfo ? (
              <Card className="border-2 border-dashed border-amber-200 bg-amber-50/50 rounded-[2.5rem] p-12 text-center">
                 <AlertCircle className="w-16 h-16 text-amber-500 mx-auto mb-6" />
                 <h3 className="text-2xl font-black text-amber-900 mb-2">Predictive Intelligence Offline</h3>
                 <p className="text-amber-700 font-medium max-w-md mx-auto">
                    The clinical model hasn't been initialized. Press the Optimize AI Engine button to start training.
                 </p>
              </Card>
           ) : (
              <Card className="border-none shadow-2xl rounded-[2.5rem] overflow-hidden bg-white ring-1 ring-slate-100">
                <CardHeader className="bg-slate-50/50 p-8 border-b border-slate-100">
                   <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                         <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                            <Database className="w-6 h-6" />
                         </div>
                         <div>
                            <CardTitle className="text-xl font-black text-slate-800">ML Engine Performance</CardTitle>
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Ensemble Architecture Evaluation</p>
                         </div>
                      </div>
                   </div>
                </CardHeader>
                <CardContent className="p-8 space-y-8">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                     <div className="col-span-1 md:col-span-2 p-8 bg-emerald-600 rounded-[2rem] text-white shadow-xl shadow-emerald-600/20 relative group overflow-hidden">
                        <TrendingUp className="absolute right-[-20px] bottom-[-20px] w-48 h-48 text-white/10 group-hover:scale-110 transition-transform duration-700" />
                        <p className="text-[10px] font-black uppercase tracking-[0.3em] opacity-80 mb-2">Primary Scoring Model</p>
                        <h4 className="text-3xl font-black mb-6">{modelInfo.best_model}</h4>
                        <div className="flex items-end justify-between">
                           <div>
                              <p className="text-4xl font-black leading-none">{modelInfo.best_auc?.toFixed(4)}</p>
                              <p className="text-[10px] font-black uppercase opacity-70">ROC AUC Metric</p>
                           </div>
                           <div className="px-4 py-2 bg-white/20 rounded-xl text-[10px] font-black uppercase border border-white/20">Active Deployment</div>
                        </div>
                     </div>
                     <div className="p-8 bg-slate-900 rounded-[2rem] text-white flex flex-col justify-between shadow-xl">
                        <Zap className="text-yellow-400 w-10 h-10 mb-4" />
                        <div>
                           <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Total Features</p>
                           <p className="text-4xl font-black">24 <span className="text-sm font-bold text-slate-600">INPUTS</span></p>
                        </div>
                     </div>
                  </div>

                  {/* Feature Importance List */}
                  <div>
                    <h5 className="text-sm font-black text-slate-800 uppercase tracking-widest mb-6">Dominant Data Drivers (Top 6)</h5>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6 text-sm">
                       {Object.entries(modelInfo.feature_importance).slice(0, 6).map(([feature, importance], idx) => (
                          <div key={idx} className="space-y-2">
                             <div className="flex justify-between items-center">
                                <span className="font-bold text-slate-600 truncate max-w-[150px] uppercase text-[10px] tracking-tight">{feature.replace(/_/g, ' ')}</span>
                                <span className="font-black text-teal-600">{(importance * 100).toFixed(1)}%</span>
                             </div>
                             <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div className="h-full bg-teal-500 rounded-full transition-all duration-1000" style={{ width: `${importance * 100}%` }} />
                             </div>
                          </div>
                       ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
           )}
        </div>

        {/* Patient Analytics (Right) */}
        <div className="xl:col-span-4 space-y-6">
           <Card className="border-none shadow-xl rounded-[2.5rem] bg-indigo-900 text-white overflow-hidden h-full">
              <CardHeader className="p-8 pb-4">
                 <Users className="w-10 h-10 text-indigo-400 mb-4" />
                 <CardTitle className="text-2xl font-black tracking-tight">Clinical Census</CardTitle>
                 <p className="text-indigo-300 text-sm font-medium">Population distribution by predicted LOS</p>
              </CardHeader>
              <CardContent className="p-8 space-y-8">
                 <div className="flex items-center justify-between p-6 bg-white/10 rounded-3xl border border-white/10">
                    <div>
                       <p className="text-[10px] font-black text-indigo-300 uppercase tracking-widest">Total Monitored</p>
                       <p className="text-4xl font-black">{stats?.total_patients || '0'}</p>
                    </div>
                    <Activity className="w-12 h-12 text-white/10" />
                 </div>

                 <div className="space-y-6">
                    <div className="space-y-3">
                       <div className="flex justify-between items-center text-xs font-black uppercase">
                          <span className="text-emerald-400">Short Stay</span>
                          <span>{stats?.short_stay_count || 0} Patients</span>
                       </div>
                       <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-400" style={{ width: `${(stats?.short_stay_count / stats?.total_patients) * 100 || 0}%` }} />
                       </div>
                    </div>

                    <div className="space-y-3">
                       <div className="flex justify-between items-center text-xs font-black uppercase">
                          <span className="text-amber-400">Long Stay</span>
                          <span>{stats?.long_stay_count || 0} Patients</span>
                       </div>
                       <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-400" style={{ width: `${(stats?.long_stay_count / stats?.total_patients) * 100 || 0}%` }} />
                       </div>
                    </div>
                 </div>

                 <div className="pt-8 border-t border-white/10">
                    <p className="text-[10px] text-indigo-400 font-bold uppercase tracking-widest mb-2">Automated Optimization</p>
                    <p className="text-xs font-medium text-indigo-200 leading-relaxed italic">
                       "Resource allocation is automatically suggesting discharges for patients in the emerald quantile."
                    </p>
                 </div>
              </CardContent>
           </Card>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
