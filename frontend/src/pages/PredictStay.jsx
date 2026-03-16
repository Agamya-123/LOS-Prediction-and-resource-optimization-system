import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Activity, TrendingUp, TrendingDown } from 'lucide-react';

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
    Department: 'Cardiology',
    Insurance_Type: 'Private',
    Num_Comorbidities: 1,
    Visitors_Count: 2,
    Blood_Sugar_Level: 120,
    Admission_Deposit: 5000,
    Diagnosis: '',
    Severity_Score: 2,
    Ward_Type: 'General'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);

    try {
      // prepare payload matching backend PatientInput schema
      const { patient_name, ...patientData } = formData;

      const payload = {
        patient_input: patientData,
        patient_name: patient_name
      };

      // The backend expects: POST /patients?patient_name=... body=PatientInput
      const response = await axios.post(
        `${API}/patients?patient_name=${encodeURIComponent(patient_name)}`,
        patientData
      );
      setPrediction(response.data);
      toast.success('Prediction completed and patient admitted!');
    } catch (error) {
      console.error(error);
      let errorMessage = 'Prediction failed';
      if (error.response?.data?.detail) {
        if (Array.isArray(error.response.data.detail)) {
          errorMessage = error.response.data.detail.map(e => e.msg).join(', ');
        } else {
          errorMessage = error.response.data.detail;
        }
      }
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field, value) => {
    setFormData(prev => {
      const newData = { ...prev, [field]: value };

      // Reset diagnosis if department changes
      if (field === 'Department') {
        newData.Diagnosis = '';
      }

      return newData;
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-900 tracking-tight">Predict Patient Stay</h1>
        <p className="text-base text-slate-600 mt-1">Enter patient details based on Indian Hospital Data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Patient Information</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">

                {/* Patient Name */}
                <div>
                  <Label htmlFor="patient_name">Patient Name *</Label>
                  <Input
                    id="patient_name"
                    value={formData.patient_name}
                    onChange={(e) => updateField('patient_name', e.target.value)}
                    required
                    placeholder="Enter patient name"
                  />
                </div>

                {/* Age & Gender */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Age">Age</Label>
                    <Input
                      id="Age"
                      type="number"
                      value={formData.Age}
                      onChange={(e) => updateField('Age', parseInt(e.target.value))}
                      min="0"
                      max="120"
                    />
                  </div>
                  <div>
                    <Label htmlFor="Gender">Gender</Label>
                    <Select value={formData.Gender} onValueChange={(v) => updateField('Gender', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Admission Type & Department */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Admission_Type">Admission Type</Label>
                    <Select value={formData.Admission_Type} onValueChange={(v) => updateField('Admission_Type', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Emergency">Emergency</SelectItem>
                        <SelectItem value="Urgent">Urgent</SelectItem>
                        <SelectItem value="Elective">Elective</SelectItem>
                        <SelectItem value="Trauma">Trauma</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="Department">Department</Label>
                    <Select value={formData.Department} onValueChange={(v) => updateField('Department', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Cardiology">Cardiology</SelectItem>
                        <SelectItem value="Neurology">Neurology</SelectItem>
                        <SelectItem value="Orthopedics">Orthopedics</SelectItem>
                        <SelectItem value="Gynecology">Gynecology</SelectItem>
                        <SelectItem value="Oncology">Oncology</SelectItem>
                        <SelectItem value="Pediatrics">Pediatrics</SelectItem>
                        <SelectItem value="Gastroenterology">Gastroenterology</SelectItem>
                        <SelectItem value="Pulmonology">Pulmonology</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Insurance & Ward Type */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Insurance_Type">Insurance Type</Label>
                    <Select value={formData.Insurance_Type} onValueChange={(v) => updateField('Insurance_Type', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Private">Private</SelectItem>
                        <SelectItem value="Medicare">Medicare</SelectItem>
                        <SelectItem value="Medicaid">Medicaid</SelectItem>
                        <SelectItem value="Uninsured">Uninsured</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="Ward_Type">Ward Type</Label>
                    <Select value={formData.Ward_Type} onValueChange={(v) => updateField('Ward_Type', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="General">General</SelectItem>
                        <SelectItem value="Semi-Private">Semi-Private</SelectItem>
                        <SelectItem value="Private">Private</SelectItem>
                        <SelectItem value="ICU">ICU</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Diagnosis & Severity */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Diagnosis">Diagnosis</Label>
                    <Select value={formData.Diagnosis} onValueChange={(v) => updateField('Diagnosis', v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select Diagnosis" />
                      </SelectTrigger>
                      <SelectContent className="max-h-[200px]">
                        {(DEPARTMENT_DIAGNOSES[formData.Department] || []).map((diag) => (
                          <SelectItem key={diag} value={diag}>{diag}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="Severity_Score">Severity Score (1-5)</Label>
                    <Input
                      id="Severity_Score"
                      type="number"
                      value={formData.Severity_Score}
                      onChange={(e) => updateField('Severity_Score', parseInt(e.target.value))}
                      min="1"
                      max="5"
                    />
                  </div>
                </div>

                {/* Comorbidities & Visitors */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Num_Comorbidities">Num Comorbidities</Label>
                    <Input
                      id="Num_Comorbidities"
                      type="number"
                      value={formData.Num_Comorbidities}
                      onChange={(e) => updateField('Num_Comorbidities', parseInt(e.target.value))}
                      min="0"
                    />
                  </div>
                  <div>
                    <Label htmlFor="Visitors_Count">Visitors Count</Label>
                    <Input
                      id="Visitors_Count"
                      type="number"
                      value={formData.Visitors_Count}
                      onChange={(e) => updateField('Visitors_Count', parseInt(e.target.value))}
                      min="0"
                    />
                  </div>
                </div>

                {/* Blood Sugar & Deposit */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Blood_Sugar_Level">Blood Sugar Level</Label>
                    <Input
                      id="Blood_Sugar_Level"
                      type="number"
                      value={formData.Blood_Sugar_Level}
                      onChange={(e) => updateField('Blood_Sugar_Level', parseInt(e.target.value))}
                      min="0"
                    />
                  </div>
                  <div>
                    <Label htmlFor="Admission_Deposit">Admission Deposit</Label>
                    <Input
                      id="Admission_Deposit"
                      type="number"
                      value={formData.Admission_Deposit}
                      onChange={(e) => updateField('Admission_Deposit', parseInt(e.target.value))}
                      min="0"
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-teal-600 hover:bg-teal-700"
                  disabled={loading}
                >
                  {loading ? 'Processing...' : 'Predict & Admit Patient'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Prediction Result */}
        <div>
          {prediction && (
            <Card className="sticky top-24">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center text-base">
                  <Activity className="w-5 h-5 mr-2" />
                  Prediction Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">

                {/* ── Section 1: Main Prediction Hero ── */}
                <div className={`p-5 rounded-lg border-2 text-center ${prediction.prediction === 0
                  ? 'bg-emerald-50 border-emerald-300'
                  : 'bg-amber-50 border-amber-300'
                  }`}>
                  <div className="flex justify-center mb-2">
                    {prediction.prediction === 0 ? (
                      <TrendingDown className="w-10 h-10 text-emerald-600" />
                    ) : (
                      <TrendingUp className="w-10 h-10 text-amber-600" />
                    )}
                  </div>
                  <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Predicted Stay</p>
                  <p className={`text-xl font-bold mt-1 ${prediction.prediction === 0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                    {prediction.prediction_label}
                  </p>
                  <p className="text-2xl font-mono font-bold mt-2 text-slate-800">
                    {(prediction.confidence * 100).toFixed(1)}%
                    <span className="text-xs font-normal text-slate-500 ml-1">confidence</span>
                  </p>
                </div>

                {/* ── Section 2: Quick Stats Row ── */}
                <div className="grid grid-cols-2 gap-2">
                  {prediction.bed_number && (
                    <div className="p-3 bg-teal-50 rounded-lg border border-teal-200 text-center">
                      <p className="text-xs text-teal-600 font-medium">Assigned Bed</p>
                      <p className="text-lg font-bold text-teal-800 font-mono">#{prediction.bed_number}</p>
                    </div>
                  )}
                  <div className="p-3 bg-slate-50 rounded-lg border text-center" style={{ gridColumn: prediction.bed_number ? 'auto' : '1 / -1' }}>
                    <p className="text-xs text-slate-500 font-medium mb-1">Probability</p>
                    <div className="flex items-center gap-1">
                      <div className="flex-1">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${prediction.probabilities.short_stay * 100}%` }} />
                        </div>
                        <p className="text-[10px] text-slate-500 mt-0.5">Short {(prediction.probabilities.short_stay * 100).toFixed(0)}%</p>
                      </div>
                      <div className="flex-1">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500 rounded-full" style={{ width: `${prediction.probabilities.long_stay * 100}%` }} />
                        </div>
                        <p className="text-[10px] text-slate-500 mt-0.5">Long {(prediction.probabilities.long_stay * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── Section 3: Anomaly Alert (if triggered) ── */}
                {prediction.is_anomaly && (
                  <div className="p-3 bg-red-50 rounded-lg border-2 border-red-300 shadow-sm animate-pulse">
                    <p className="text-sm font-bold text-red-800 flex items-center mb-1">
                      <Activity className="w-4 h-4 mr-2" />
                      ⚠️ Clinical Anomaly Detected
                    </p>
                    <p className="text-xs text-red-700 leading-relaxed">
                      This patient's clinical presentation is highly unusual (Top 5% outlier). Review data for potential entry errors or rare clinical conditions.
                    </p>
                  </div>
                )}

                {/* ── Section 4: XAI SHAP Visual Explanation ── */}
                {prediction.shap_explanation && Object.keys(prediction.shap_explanation).length > 0 && (
                  <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                    <p className="text-sm font-semibold text-purple-800 mb-1">
                      🧠 XAI: SHAP Feature Explanation
                    </p>
                    <p className="text-[10px] text-purple-500 mb-3">
                      Game Theory mathematics — each feature's exact contribution to this prediction
                    </p>
                    <div className="space-y-1.5">
                      {Object.entries(prediction.shap_explanation).slice(0, 5).map(([feature, value], idx) => (
                        <div key={idx} className="flex items-center gap-1.5">
                          <span className="text-[11px] text-purple-700 w-24 truncate font-medium" title={feature}>{feature}</span>
                          <div className="flex-1 h-3 bg-purple-100 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${value > 0 ? 'bg-red-400' : 'bg-emerald-400'}`}
                              style={{ width: `${Math.min(Math.abs(value) * 150, 100)}%` }}
                            />
                          </div>
                          <span className={`text-[10px] font-mono w-14 text-right font-bold ${value > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                            {value > 0 ? '+' : ''}{value.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="flex gap-4 mt-2">
                      <span className="text-[10px] text-red-500 flex items-center gap-1"><span className="w-2 h-2 bg-red-400 rounded-full inline-block"></span> → Long Stay</span>
                      <span className="text-[10px] text-emerald-500 flex items-center gap-1"><span className="w-2 h-2 bg-emerald-400 rounded-full inline-block"></span> → Short Stay</span>
                    </div>
                  </div>
                )}

                {/* ── Section 5: Contributing Factors ── */}
                {prediction.contributing_factors && prediction.contributing_factors.length > 0 ? (
                  <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-sm font-semibold text-blue-800 mb-2">📋 Contributing Factors</p>
                    <ul className="text-xs text-blue-700 space-y-1">
                      {prediction.contributing_factors.map((factor, idx) => (
                        <li key={idx} className="flex items-start">
                          <span className="mr-1.5 text-blue-400">•</span>
                          <span>{factor}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div className="p-3 bg-slate-50 rounded-lg border text-xs text-slate-500 text-center">
                    No critical high-risk flags detected.
                  </div>
                )}

                {/* ── Section 6: AI Recommended Actions ── */}
                {prediction.recommended_actions && prediction.recommended_actions.length > 0 && (
                  <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
                    <p className="text-sm font-semibold text-indigo-800 mb-2">💊 AI Recommended Actions</p>
                    <ul className="text-xs text-indigo-700 space-y-1">
                      {prediction.recommended_actions.map((action, idx) => (
                        <li key={idx} className="flex items-start">
                          <span className="mr-1.5 text-indigo-400 font-bold">→</span>
                          <span>{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictStay;
