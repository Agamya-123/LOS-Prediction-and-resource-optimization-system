import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Activity, TrendingUp, Zap, Layout, Info, User, ClipboardList, ShieldAlert, BedDouble } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DEPARTMENT_DIAGNOSES = {
  'Cardiology': ['Heart Failure', 'Coronary Artery Disease', 'Angina'],
  'Neurology': ['Stroke', 'Epilepsy', 'Migraine'],
  'Orthopedics': ['Hip Fracture', 'Knee Replacement', 'Back Pain'],
  'Pulmonology': ['Viral Infection', 'Pneumonia', 'Bronchitis', 'COPD', 'Asthma Attack'],
  'Gastroenterology': ['Gastritis', 'Pancreatitis', 'Appendicitis'],
  'Pediatrics': ['Viral Infection', 'Asthma Attack', 'Dehydration'],
  'Oncology': ['Tumor Surgery', 'Chemotherapy', 'Radiotherapy'],
  'Gynecology': ['C-Section', 'Normal Delivery', 'Hysterectomy']
};

const PredictStay = () => {
  const [formData, setFormData] = useState({
    patient_name: '',
    Age: 45,
    Gender: 'Male',
    Admission_Type: 'Emergency',
    Department: 'Oncology',
    Insurance_Type: 'Private',
    Num_Comorbidities: 1,
    Visitors_Count: 2,
    Blood_Sugar_Level: 120,
    Admission_Deposit: 5000,
    Diagnosis: 'Tumor Surgery',
    Severity_Score: 3,
    Ward_Type: 'General'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    try {
      const { patient_name, ...patientData } = formData;
      const response = await axios.post(
        `${API}/patients?patient_name=${encodeURIComponent(patient_name)}`,
        patientData
      );
      setPrediction(response.data);
      toast.success('Clinical analysis complete!');
    } catch (error) {
       toast.error('Analysis failed. Please check server connection.');
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="max-w-[1600px] mx-auto p-4 md:p-6 space-y-4 animate-in fade-in duration-500 overflow-hidden">
      
      {/* ── Dashboard Header ── */}
      <div className="bg-slate-900 rounded-3xl p-5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-2xl border border-slate-800">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-teal-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
            <Activity className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">AI Prognosis <span className="text-teal-400">Dashboard</span></h1>
            <p className="text-slate-400 text-xs font-semibold uppercase tracking-[0.2em]">Ensemble Prediction Engine v2.4</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="hidden lg:flex gap-6 pr-6 border-r border-slate-800">
             <div className="text-center">
                <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Status</p>
                <div className="flex items-center gap-1.5">
                   <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                   <span className="text-xs text-white font-bold">Live System</span>
                </div>
             </div>
             <div className="text-center">
                <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">XAI Depth</p>
                <span className="text-xs text-teal-400 font-bold">SHAP Kernel</span>
             </div>
          </div>
          <Button 
            form="prediction-form" 
            type="submit"
            className="h-12 px-8 rounded-2xl bg-teal-500 hover:bg-teal-400 text-white font-black shadow-lg shadow-teal-500/20 transition-all hover:scale-105"
            disabled={loading}
          >
            {loading ? 'ANALYZING...' : 'GENERATE REPORT'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        
        {/* ── Left Side: Global Form (Horizontal Style) ── */}
        <div className="xl:col-span-8 space-y-4">
          <Card className="border-none shadow-xl rounded-3xl overflow-hidden ring-1 ring-slate-100 h-full">
            <CardHeader className="bg-slate-50/50 p-4 border-b border-slate-100 flex flex-row items-center">
               <User className="w-4 h-4 mr-2 text-slate-400" />
               <CardTitle className="text-sm font-bold text-slate-700">Patient Profiling</CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <form id="prediction-form" onSubmit={handleSubmit} className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4">
                <div className="col-span-2 space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Patient Identity</Label>
                  <Input 
                    value={formData.patient_name} 
                    onChange={(e) => updateField('patient_name', e.target.value)}
                    className="rounded-xl h-10 border-slate-200 bg-slate-50/50 focus:bg-white" 
                    placeholder="Search or enter name..." 
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Age (Years)</Label>
                  <Input type="number" value={formData.Age} onChange={(e) => updateField('Age', e.target.value)} className="rounded-xl h-10" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Gender</Label>
                  <Select value={formData.Gender} onValueChange={(v) => updateField('Gender', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent><SelectItem value="Male">Male</SelectItem><SelectItem value="Female">Female</SelectItem></SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Clinical Dept.</Label>
                  <Select value={formData.Department} onValueChange={(v) => updateField('Department', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.keys(DEPARTMENT_DIAGNOSES).map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Primary Diagnosis</Label>
                  <Select value={formData.Diagnosis} onValueChange={(v) => updateField('Diagnosis', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {DEPARTMENT_DIAGNOSES[formData.Department]?.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Admission Type</Label>
                  <Select value={formData.Admission_Type} onValueChange={(v) => updateField('Admission_Type', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Emergency">Emergency</SelectItem>
                      <SelectItem value="Trauma">Trauma</SelectItem>
                      <SelectItem value="Urgent">Urgent</SelectItem>
                      <SelectItem value="Elective">Elective</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Ward Environment</Label>
                  <Select value={formData.Ward_Type} onValueChange={(v) => updateField('Ward_Type', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ICU">ICU Unit</SelectItem>
                      <SelectItem value="Private">Private</SelectItem>
                      <SelectItem value="General">General</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Severity Flag (1-5)</Label>
                  <div className="flex bg-slate-100 p-0.5 rounded-xl h-10">
                     {[1,2,3,4,5].map(v => (
                        <button 
                          key={v}
                          type="button"
                          onClick={() => updateField('Severity_Score', v)}
                          className={`flex-1 rounded-lg text-xs font-black transition-all ${formData.Severity_Score === v ? 'bg-orange-500 text-white shadow-md' : 'text-slate-500 hover:text-slate-800'}`}
                        >
                          {v}
                        </button>
                     ))}
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Comorbidities</Label>
                  <Input type="number" value={formData.Num_Comorbidities} onChange={(e) => updateField('Num_Comorbidities', e.target.value)} className="rounded-xl h-10" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Deposit (INR)</Label>
                  <Input type="number" value={formData.Admission_Deposit} onChange={(e) => updateField('Admission_Deposit', e.target.value)} className="rounded-xl h-10" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-slate-400 font-bold uppercase ml-1">Insurance</Label>
                  <Select value={formData.Insurance_Type} onValueChange={(v) => updateField('Insurance_Type', v)}>
                    <SelectTrigger className="rounded-xl h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                        <SelectItem value="Private">Private</SelectItem>
                        <SelectItem value="Medicare">Govt/Medicare</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* ── Right Side: Main Result Hero ── */}
        <div className="xl:col-span-4 h-full">
           {!prediction ? (
             <div className="h-full bg-slate-50 border-2 border-dashed border-slate-200 rounded-3xl flex flex-col items-center justify-center p-8 text-center min-h-[300px]">
                <ClipboardList className="w-12 h-12 text-slate-200 mb-4" />
                <p className="text-slate-400 font-bold text-sm">PROGNOSIS ENGINE READY</p>
                <p className="text-slate-300 text-xs mt-1">Pending clinical data submission</p>
             </div>
           ) : (
             <div className="flex flex-col gap-4 h-full">
                {/* Result Hero Card */}
                <Card className={`border-none shadow-2xl rounded-3xl overflow-hidden flex-1 transition-all duration-500 ${prediction.prediction === 1 ? 'bg-orange-600' : 'bg-emerald-600'}`}>
                   <CardContent className="p-8 flex flex-col items-center justify-center text-center text-white h-full relative">
                      <Zap className="absolute top-6 right-6 w-12 h-12 text-white/10" />
                      <p className="text-xs font-black uppercase tracking-[0.3em] text-white/70 mb-2">AI Forecasted Outcome</p>
                      <h2 className="text-4xl font-black mb-4 tracking-tight drop-shadow-md">
                        {prediction.prediction_label}
                      </h2>
                      <div className="w-32 h-32 rounded-full border-4 border-white/20 flex flex-col items-center justify-center bg-white/5 backdrop-blur-md mb-4 group hover:scale-110 transition-transform">
                         <span className="text-3xl font-black">{(prediction.confidence*100).toFixed(0)}%</span>
                         <span className="text-[9px] font-bold opacity-70">CONFIDENCE</span>
                      </div>
                      <div className="flex gap-1.5">
                         <div className="px-3 py-1 bg-white/10 rounded-full border border-white/20 text-[10px] font-bold">MODEL-TRIALIZED</div>
                      </div>
                   </CardContent>
                </Card>
                
                {/* Bed Card */}
                {prediction.bed_number && (
                  <div className="bg-slate-900 rounded-3xl p-5 flex items-center justify-between border border-slate-800 shadow-xl group">
                    <div className="flex items-center gap-4">
                       <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                          <BedDouble className="w-5 h-5" />
                       </div>
                       <div>
                          <p className="text-[10px] text-slate-500 font-bold uppercase">Resource Assignment</p>
                          <p className="text-white font-black">Bed Unit #{prediction.bed_number}</p>
                       </div>
                    </div>
                    <div className="text-right">
                       <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2.5 py-1 rounded-full font-black">STANDBY</span>
                    </div>
                  </div>
                )}
             </div>
           )}
        </div>

        {/* ── Bottom Section: Deep Analysis Dashboard ── */}
        {prediction && (
          <div className="xl:col-span-12 grid grid-cols-1 md:grid-cols-12 gap-4 animate-in slide-in-from-bottom duration-700">
            
            {/* Probability Tiles */}
            <div className="md:col-span-3 space-y-4">
               <div className="bg-white p-5 rounded-3xl shadow-xl border border-slate-100 flex items-center justify-between">
                  <div>
                    <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Stay Duration</p>
                    <p className="text-lg font-black text-slate-800">Probabilities</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-slate-100" />
               </div>
               <div className="grid grid-cols-2 gap-4">
                  <div className="bg-emerald-50 p-4 rounded-3xl border border-emerald-100 text-center">
                     <p className="text-[10px] text-emerald-600 font-black uppercase mb-1">Short</p>
                     <p className="text-2xl font-black text-emerald-900">{(prediction.probabilities.short_stay * 100).toFixed(0)}%</p>
                  </div>
                  <div className="bg-orange-50 p-4 rounded-3xl border border-orange-100 text-center">
                     <p className="text-[10px] text-orange-600 font-black uppercase mb-1">Long</p>
                     <p className="text-2xl font-black text-orange-900">{(prediction.probabilities.long_stay * 100).toFixed(0)}%</p>
                  </div>
               </div>
            </div>

            {/* XAI Visualization (The Heart) */}
            <div className="md:col-span-6 bg-white rounded-3xl shadow-2xl p-6 border border-slate-100 flex flex-col justify-between">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                   <Zap className="w-5 h-5 text-purple-500" />
                   <h3 className="font-bold text-slate-800">Explainable Decision Drivers</h3>
                </div>
                <div className="flex gap-4 text-[9px] font-black">
                   <span className="text-emerald-500">← SHORTER</span>
                   <span className="text-orange-500">LONGER →</span>
                </div>
              </div>
              
              <div className="grid grid-cols-5 gap-6">
                {Object.entries(prediction.shap_explanation).slice(0, 5).map(([feature, value], idx) => (
                  <div key={idx} className="flex flex-col items-center">
                     <div className="h-28 w-4 bg-slate-50 rounded-full relative flex flex-col justify-center overflow-hidden mb-2">
                        <div className="absolute left-0 right-0 h-px bg-slate-300 z-10" style={{ top: '50%' }} />
                        {value > 0 ? (
                           <div className="absolute bg-orange-400 w-full rounded-t-sm transition-all duration-1000 bottom-1/2" style={{ height: `${Math.min(value * 200, 50)}%` }} />
                        ) : (
                           <div className="absolute bg-emerald-400 w-full rounded-b-sm transition-all duration-1000 top-1/2" style={{ height: `${Math.min(Math.abs(value) * 200, 50)}%` }} />
                        )}
                     </div>
                     <p className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter text-center h-4 overflow-hidden truncate w-full" title={feature}>{feature.replace(/_/g, ' ')}</p>
                     <p className={`text-[10px] font-black ${value > 0 ? 'text-orange-600' : 'text-emerald-600'}`}>{value > 0 ? '+' : ''}{value.toFixed(2)}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence & Anomaly */}
            <div className="md:col-span-3 space-y-4">
               {prediction.is_anomaly ? (
                  <div className="bg-red-500 rounded-3xl p-5 text-white shadow-lg shadow-red-500/20 flex flex-col justify-center h-full">
                     <div className="flex items-center gap-3 mb-2">
                        <ShieldAlert className="w-6 h-6" />
                        <span className="font-black text-sm uppercase">Medical Outlier</span>
                     </div>
                     <p className="text-[11px] font-bold leading-relaxed opacity-90">Rare clinical pattern detected. Prediction trust is weighted lower by the system. Clinical review mandatory.</p>
                  </div>
                ) : (
                  <div className="bg-slate-50 rounded-3xl p-5 flex flex-col justify-start h-full border border-slate-200">
                    <p className="text-[10px] text-slate-400 font-black uppercase mb-3 px-1">System Observations</p>
                    <div className="space-y-3 overflow-y-auto custom-scrollbar max-h-[120px] pr-1">
                      {prediction.contributing_factors.map((fact, i) => (
                        <div key={i} className="flex gap-2.5 items-start">
                           <div className="w-1.5 h-1.5 rounded-full bg-slate-300 mt-1.5 shrink-0" />
                           <p className="text-[10px] font-bold text-slate-600 leading-tight">{fact}</p>
                        </div>
                      ))}
                    </div>
                  </div>
               )}
            </div>

          </div>
        )}

      </div>
    </div>
  );
};

export default PredictStay;
