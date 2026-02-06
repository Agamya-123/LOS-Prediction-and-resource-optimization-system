import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import { UserMinus, Search } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const PatientRecords = () => {
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPatients();
  }, []);

  useEffect(() => {
    const filtered = patients.filter(patient =>
      patient.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredPatients(filtered);
  }, [searchTerm, patients]);

  const fetchPatients = async () => {
    try {
      const response = await axios.get(`${API}/patients`);
      setPatients(response.data);
      setFilteredPatients(response.data);
    } catch (error) {
      console.error('Error fetching patients:', error);
      toast.error('Failed to load patient records');
    } finally {
      setLoading(false);
    }
  };

  const dischargePatient = async (patientId) => {
    try {
      await axios.delete(`${API}/patients/${patientId}`);
      toast.success('Patient discharged successfully');
      fetchPatients();
    } catch (error) {
      console.error(error);
      toast.error('Failed to discharge patient');
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading patients...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-900 tracking-tight">Patient Records</h1>
          <p className="text-base text-slate-600 mt-1">{patients.length} total patients</p>
        </div>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
            <Input
              data-testid="search-patients"
              placeholder="Search by patient name or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardContent>
      </Card>

      {/* Patient List */}
      {filteredPatients.length === 0 ? (
        <Card>
          <CardContent className="p-12 text-center">
            <p className="text-slate-500">No patients found</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredPatients.map((patient) => (
            <Card key={patient.id} data-testid={`patient-card-${patient.id}`} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <h3 className="text-xl font-bold text-slate-900">{patient.patient_name}</h3>
                      <Badge
                        className={patient.prediction === 0
                          ? 'bg-emerald-100 text-emerald-700 border-emerald-200'
                          : 'bg-amber-100 text-amber-700 border-amber-200'
                        }
                      >
                        {patient.prediction_label}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-slate-500">Patient ID</p>
                        <p className="font-mono text-slate-900">{patient.id.slice(0, 8)}...</p>
                      </div>
                      <div>
                        <p className="text-slate-500">Age / Gender</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Age ?? 'N/A'} / {patient.patient_data?.Gender ?? 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Bed Number</p>
                        <p className="font-mono font-bold text-teal-600">
                          {patient.bed_number ? `Bed #${patient.bed_number}` : 'Not Assigned'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Ward Type</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Ward_Type ?? 'General'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Department</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Department ?? 'General'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Diagnosis</p>
                        <p className="font-medium text-slate-900truncate" title={patient.patient_data?.Diagnosis}>
                          {patient.patient_data?.Diagnosis ?? 'Unknown'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Severity Score</p>
                        <div className="flex items-center">
                          <span className={`font-bold ${(patient.patient_data?.Severity_Score || 0) > 3 ? 'text-red-600' : 'text-slate-900'
                            }`}>
                            {patient.patient_data?.Severity_Score ?? 'N/A'}
                          </span>
                          <span className="text-slate-400 text-xs ml-1">/ 5</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-slate-500">Admission Type</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Admission_Type ?? 'Unknown'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Insurance</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Insurance_Type ?? 'None'}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Comorbidities</p>
                        <p className="font-medium text-slate-900">
                          {patient.patient_data?.Num_Comorbidities ?? 0}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Deposit</p>
                        <p className="font-mono text-slate-900">
                          ₹{patient.patient_data?.Admission_Deposit?.toLocaleString() ?? 0}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Predicted Discharge</p>
                        <p className="font-medium text-slate-900">{patient.predicted_discharge ?? 'Pending'}</p>
                      </div>
                      <div>
                        <p className="text-slate-500">Confidence</p>
                        <p className="font-mono font-medium text-slate-900">
                          {((patient.confidence || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <Button
                    data-testid={`discharge-button-${patient.id}`}
                    onClick={() => dischargePatient(patient.id)}
                    variant="destructive"
                    size="sm"
                    className="ml-4"
                  >
                    <UserMinus className="w-4 h-4 mr-2" />
                    Discharge
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default PatientRecords;
