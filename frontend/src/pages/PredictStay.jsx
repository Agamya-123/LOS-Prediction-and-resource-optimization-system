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

const PredictStay = () => {
  const [formData, setFormData] = useState({
    patient_name: '',
    Age: 50,
    Gender: 'Male',
    Admission_Type: 'Emergency',
    Department: 'Cardiology',
    Comorbidity: 'None',
    Procedures: 0
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
    setFormData(prev => ({ ...prev, [field]: value }));
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
                        <SelectItem value="Referral">Referral</SelectItem>
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
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Comorbidity & Procedures */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="Comorbidity">Comorbidity</Label>
                    <Select value={formData.Comorbidity} onValueChange={(v) => updateField('Comorbidity', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="None">None</SelectItem>
                        <SelectItem value="Diabetes">Diabetes</SelectItem>
                        <SelectItem value="Hypertension">Hypertension</SelectItem>
                        <SelectItem value="Heart Disease">Heart Disease</SelectItem>
                        <SelectItem value="Cancer">Cancer</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="Procedures">Number of Procedures</Label>
                    <Input
                      id="Procedures"
                      type="number"
                      value={formData.Procedures}
                      onChange={(e) => updateField('Procedures', parseInt(e.target.value))}
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
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Activity className="w-5 h-5 mr-2" />
                  Prediction Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className={`p-6 rounded-lg border-2 ${
                  prediction.prediction === 0
                    ? 'bg-emerald-50 border-emerald-300'
                    : 'bg-amber-50 border-amber-300'
                }`}>
                  {prediction.prediction === 0 ? (
                    <TrendingDown className="w-12 h-12 text-emerald-600 mb-3" />
                  ) : (
                    <TrendingUp className="w-12 h-12 text-amber-600 mb-3" />
                  )}
                  <p className="text-sm font-medium text-slate-600">Predicted Stay</p>
                  <p className={`text-2xl font-bold mt-1 ${
                    prediction.prediction === 0 ? 'text-emerald-700' : 'text-amber-700'
                  }`}>
                    {prediction.prediction_label}
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded border">
                    <span className="text-sm text-slate-600">Confidence</span>
                    <span className="font-mono font-bold text-slate-900">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {prediction.bed_number && (
                    <div className="flex justify-between items-center p-3 bg-teal-50 rounded border border-teal-200">
                      <span className="text-sm text-teal-700">Assigned Bed</span>
                      <span className="font-mono font-bold text-teal-900">
                        Bed #{prediction.bed_number}
                      </span>
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t">
                  <p className="text-xs text-slate-500 mb-2">Probabilities</p>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span>Short Stay</span>
                        <span className="font-mono">{(prediction.probabilities.short_stay * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full">
                        <div
                          className="h-full bg-emerald-500 rounded-full"
                          style={{ width: `${prediction.probabilities.short_stay * 100}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span>Long Stay</span>
                        <span className="font-mono">{(prediction.probabilities.long_stay * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full">
                        <div
                          className="h-full bg-amber-500 rounded-full"
                          style={{ width: `${prediction.probabilities.long_stay * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictStay;
