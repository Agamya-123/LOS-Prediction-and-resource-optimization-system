import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import { UserMinus, Search, User, Calendar, BedDouble, ShieldCheck, CreditCard, Activity, ClipboardList, ShieldAlert, Zap, TrendingUp } from 'lucide-react';
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
      toast.success('Electronic Health Record archived & Patient Discharged');
      fetchPatients();
    } catch (error) {
      toast.error('Failed to process discharge');
    }
  };

  if (loading) {
    return (
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <Search className="w-10 h-10 text-slate-200 animate-pulse mb-4" />
          <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">Scanning Patient Database...</p>
        </div>
      );
  }

  return (
    <div className="max-w-[1600px] mx-auto space-y-6 animate-in fade-in duration-500">
      
      {/* ── Dynamic Header ── */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 bg-white p-8 rounded-[2.5rem] shadow-xl border border-slate-100">
        <div className="space-y-1">
          <h1 className="text-4xl font-black text-slate-800 tracking-tight">EHR <span className="text-indigo-600">Archive</span></h1>
          <p className="text-slate-500 font-medium flex items-center">
            <ShieldCheck className="w-4 h-4 mr-2 text-emerald-500" />
            {patients.length} Active clinical records under monitoring
          </p>
        </div>

        <div className="relative w-full md:w-[400px] group">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
          <Input
            placeholder="Search by name, ID or diagnosis..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-12 h-14 rounded-2xl border-slate-200 bg-slate-50/50 focus:bg-white focus:ring-2 focus:ring-indigo-500/20 transition-all font-medium text-slate-700"
          />
        </div>
      </div>

      {/* ── Results Canvas ── */}
      {filteredPatients.length === 0 ? (
        <div className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-[2.5rem] p-24 text-center">
           <div className="w-20 h-20 bg-white rounded-3xl shadow-lg flex items-center justify-center mx-auto mb-6">
              <Search className="w-10 h-10 text-slate-200" />
           </div>
           <h3 className="text-xl font-black text-slate-800 mb-2">No Records Found</h3>
           <p className="text-slate-500 max-w-xs mx-auto text-sm">We couldn't find any patient records matching your search criteria.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredPatients.map((patient) => (
            <Card key={patient.id} className="border-none shadow-lg rounded-[2rem] overflow-hidden group hover:shadow-2xl transition-all duration-500 ring-1 ring-slate-100/50">
              <CardContent className="p-0">
                <div className="flex flex-col lg:flex-row">
                   
                   {/* Left Identifier Panel */}
                   <div className="lg:w-72 p-8 bg-slate-50 border-r border-slate-100 flex flex-col justify-between items-center text-center">
                      <div className="w-20 h-20 bg-white rounded-[2rem] shadow-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-500">
                         <User className="w-10 h-10 text-slate-300" />
                      </div>
                      <div className="space-y-1">
                         <h3 className="text-lg font-black text-slate-800 leading-tight">{patient.patient_name}</h3>
                         <p className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">{patient.id.slice(0, 12)}</p>
                      </div>
                      <div className="mt-6 w-full">
                         <Badge 
                           className={`w-full justify-center h-10 rounded-xl border-none text-[10px] font-black uppercase tracking-widest shadow-sm ${
                             patient.prediction === 0 
                               ? 'bg-emerald-500 text-white shadow-emerald-500/20' 
                               : 'bg-orange-500 text-white shadow-orange-500/20'
                           }`}
                         >
                           {patient.prediction_label}
                         </Badge>
                      </div>
                   </div>

                   {/* Main Clinical Data Grid */}
                   <div className="flex-1 p-8">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-6">
                         {[
                           { label: 'Demographics', val: `${patient.patient_data?.Age || '?'}Y / ${patient.patient_data?.Gender?.charAt(0) || '?'}`, icon: Calendar },
                           { label: 'Location', val: patient.bed_number ? `Unit #${patient.bed_number}` : 'Standby', icon: BedDouble },
                           { label: 'Ward Type', val: patient.patient_data?.Ward_Type || 'General', icon: ShieldCheck },
                           { label: 'Department', val: patient.patient_data?.Department || 'N/A', icon: Activity },
                           { label: 'Diagnosis', val: patient.patient_data?.Diagnosis || 'Routine', icon: ClipboardList, full: true },
                           { label: 'Risk Factor', val: `${patient.patient_data?.Severity_Score || 0}/5 Score`, icon: ShieldAlert },
                           { label: 'Trust Index', val: `${((patient.confidence || 0) * 100).toFixed(1)}% Acc.`, icon: Zap },
                           { label: 'Stay Outlook', val: patient.predicted_discharge || 'Pending', icon: TrendingUp },
                         ].map((item, idx) => {
                            const Icon = item.icon;
                            return (
                              <div key={idx} className={`${item.full ? 'col-span-2' : ''} space-y-1.5`}>
                                <div className="flex items-center gap-2">
                                   <Icon className="w-3 h-3 text-slate-300" />
                                   <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{item.label}</p>
                                </div>
                                <p className="text-sm font-bold text-slate-700 truncate" title={item.val}>{item.val}</p>
                              </div>
                            );
                          })}
                      </div>

                      <div className="mt-8 pt-8 border-t border-slate-100 flex items-center justify-between">
                         <div className="flex items-center gap-6">
                            <div className="flex items-center gap-2">
                               <CreditCard className="w-4 h-4 text-slate-300" />
                               <span className="text-[11px] font-bold text-slate-500">Deposit: ₹{patient.patient_data?.Admission_Deposit?.toLocaleString() || 0}</span>
                            </div>
                            <div className="w-px h-4 bg-slate-100" />
                            <div className="flex items-center gap-2">
                               <UserMinus className="w-4 h-4 text-slate-300" />
                               <span className="text-[11px] font-bold text-slate-500">Comorbidities: {patient.patient_data?.Num_Comorbidities || 0}</span>
                            </div>
                         </div>
                         
                         <Button
                           onClick={() => dischargePatient(patient.id)}
                           variant="outline"
                           className="h-10 px-6 rounded-xl border-slate-200 text-rose-600 font-black hover:bg-rose-50 hover:border-rose-100 transition-all text-[10px] uppercase tracking-widest"
                         >
                           Archive Record & Discharge
                         </Button>
                      </div>
                   </div>

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
