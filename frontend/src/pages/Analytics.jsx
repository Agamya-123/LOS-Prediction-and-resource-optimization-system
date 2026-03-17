import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { Activity, PieChart as PieIcon, BarChart3, Zap, Globe, TrendingUp, Info } from 'lucide-react';

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
    return (
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <BarChart3 className="w-10 h-10 text-teal-400 animate-bounce mb-4" />
          <p className="text-xs font-black text-slate-400 uppercase tracking-widest">Compiling Analytics Suite...</p>
        </div>
      );
  }

  const bedStatusData = [
    { name: 'Available', value: stats?.available_beds || 0, color: '#10B981', gradient: 'from-emerald-400 to-emerald-600' },
    { name: 'Occupied', value: stats?.occupied_beds || 0, color: '#EF4444', gradient: 'from-rose-400 to-rose-600' },
    { name: 'Cleaning', value: stats?.cleaning_beds || 0, color: '#F59E0B', gradient: 'from-amber-400 to-amber-600' }
  ];

  const predictionData = [
    { name: 'Short Stay', count: stats?.short_stay_count || 0, fill: '#10B981' },
    { name: 'Long Stay', count: stats?.long_stay_count || 0, fill: '#6366F1' }
  ];

  const modelComparisonData = modelInfo?.model_comparison
    ? Object.entries(modelInfo.model_comparison).map(([name, metrics]) => ({
        name: name.replace('Classifier', ''),
        auc: metrics.auc,
        accuracy: metrics.accuracy
      }))
    : [];

  return (
    <div className="max-w-[1600px] mx-auto space-y-6 animate-in fade-in duration-700">
      
      {/* ── Analytical Header ── */}
      <div className="bg-white rounded-[2.5rem] p-8 md:p-10 shadow-xl border border-slate-100 flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="space-y-2">
           <div className="flex items-center gap-2 mb-2">
              <span className="w-3 h-3 rounded-full bg-teal-500 animate-pulse" />
              <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Business Intelligence</span>
           </div>
           <h1 className="text-4xl font-black text-slate-800 tracking-tight">Statistical <span className="text-teal-600">Forecasting</span></h1>
           <p className="text-slate-500 font-medium max-w-lg leading-relaxed text-sm">
             Deep-dive into hospital operational metrics, resource distribution, and cross-model performance validation.
           </p>
        </div>
        <div className="flex gap-4">
           <div className="text-right">
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none mb-1">Total Census</p>
              <p className="text-3xl font-black text-slate-800 tracking-tighter">{stats?.total_patients || 0}</p>
           </div>
           <div className="w-px h-12 bg-slate-100 hidden md:block" />
           <div className="hidden md:block text-right">
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none mb-1">Bed Flux</p>
              <p className="text-3xl font-black text-emerald-500 tracking-tighter">Stability</p>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Resource Distribution (Pie) */}
        <Card className="lg:col-span-5 border-none shadow-xl rounded-[2.5rem] overflow-hidden group">
          <CardHeader className="p-8 pb-0">
             <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
                   <PieIcon className="w-4 h-4" />
                </div>
                <CardTitle className="text-lg font-black text-slate-800">Bed Status Inventory</CardTitle>
             </div>
          </CardHeader>
          <CardContent className="p-8">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={bedStatusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {bedStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ borderRadius: '20px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', fontWeight: 'bold', fontSize: '12px' }}
                />
                <Legend iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-3 gap-4 mt-4">
               {bedStatusData.map((item, i) => (
                  <div key={i} className={`p-4 rounded-2xl bg-slate-50 border border-slate-100 text-center group-hover:bg-white group-hover:shadow-lg transition-all`}>
                      <p className="text-[8px] font-black text-slate-400 uppercase mb-1">{item.name}</p>
                      <p className="text-xl font-black text-slate-800">{item.value}</p>
                  </div>
               ))}
            </div>
          </CardContent>
        </Card>

        {/* Prediction Spread (Bar) */}
        <Card className="lg:col-span-7 border-none shadow-xl rounded-[2.5rem] overflow-hidden">
          <CardHeader className="p-8 pb-0">
             <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                   <TrendingUp className="w-4 h-4" />
                </div>
                <CardTitle className="text-lg font-black text-slate-800">Stay Duration Outlook</CardTitle>
             </div>
          </CardHeader>
          <CardContent className="p-8">
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={predictionData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis 
                  dataKey="name" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: '#94a3b8', fontWeight: 900, fontSize: 10 }}
                  dy={10}
                />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontWeight: 900, fontSize: 10 }} />
                <Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: '15px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                <Bar dataKey="count" radius={[15, 15, 15, 15]} barSize={60}>
                  {predictionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cross-Model Benchmarking (Wide Bar) */}
        {modelComparisonData.length > 0 && (
          <Card className="lg:col-span-12 border-none shadow-xl rounded-[2.5rem] overflow-hidden">
            <CardHeader className="bg-slate-50/50 p-8 border-b border-slate-100">
               <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5 text-teal-600" />
                  <div>
                     <CardTitle className="text-xl font-black text-slate-800">Cross-Model Benchmarking</CardTitle>
                     <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Comparative validation of architecture performance</p>
                  </div>
               </div>
            </CardHeader>
            <CardContent className="p-8">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelComparisonData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" domain={[0, 1]} hide />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: '#475569', fontWeight: 900, fontSize: 11 }}
                    width={150}
                  />
                  <Tooltip cursor={{ fill: '#f8fafc' }} />
                  <Bar dataKey="auc" fill="#0D9488" name="ROC AUC" radius={[0, 10, 10, 0]} barSize={20} />
                  <Bar dataKey="accuracy" fill="#6366F1" name="ACCURACY" radius={[0, 10, 10, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Intelligence Matrix (Feature Importance) */}
        <Card className="lg:col-span-12 border-none shadow-2xl rounded-[3rem] bg-slate-950 text-white overflow-hidden">
          <CardContent className="p-12 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-12 opacity-10">
               <Zap className="w-64 h-64 text-teal-400" />
            </div>
            <div className="relative z-10 grid grid-cols-1 xl:grid-cols-2 gap-16">
               <div className="space-y-6">
                  <div className="inline-flex items-center gap-3 px-4 py-2 bg-teal-500/10 rounded-full border border-teal-500/20">
                     <Activity className="w-4 h-4 text-teal-400" />
                     <span className="text-[10px] font-black uppercase tracking-[0.2em] text-teal-400">Decision Intelligence</span>
                  </div>
                  <h3 className="text-4xl font-black tracking-tight leading-tight">What factors <br/> drive clinical <span className="text-teal-400 underline decoration-teal-400/30 decoration-8 underline-offset-8">outcomes?</span></h3>
                  <p className="text-slate-400 font-medium leading-relaxed max-w-md">
                    Our AI weighs dozens of clinical markers. This matrix represents the mathematical priority given to specific patient attributes during stay forecasting.
                  </p>
               </div>

               <div className="space-y-6">
                 {modelInfo?.feature_importance && Object.entries(modelInfo.feature_importance)
                   .slice(0, 6)
                   .map(([feature, importance], idx) => (
                     <div key={idx} className="space-y-2 group">
                        <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest text-slate-500 group-hover:text-teal-400 transition-colors">
                           <span>{feature.replace(/_/g, ' ')}</span>
                           <span>{Math.round(importance * 100)}% WEIGHT</span>
                        </div>
                        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                           <div className="h-full bg-teal-500 shadow-[0_0_10px_rgba(20,184,166,0.3)] transition-all duration-1000" style={{ width: `${importance * 100}%` }} />
                        </div>
                     </div>
                   ))}
               </div>
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  );
};

export default Analytics;
