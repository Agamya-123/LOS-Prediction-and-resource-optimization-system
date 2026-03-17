import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Bed as BedIcon, Activity, Wind, Loader2, Layout, Info, UserCheck, ShieldAlert } from 'lucide-react';

import { Badge } from '@/components/ui/badge';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const BedManagement = () => {
  const [beds, setBeds] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBeds();
    const interval = setInterval(fetchBeds, 5000); // Live sync
    return () => clearInterval(interval);
  }, []);

  const fetchBeds = async () => {
    try {
      const response = await axios.get(`${API}/beds`);
      setBeds(response.data);
    } catch (error) {
      console.error('Error fetching beds:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateBedStatus = async (bedNumber, status) => {
    try {
      await axios.patch(`${API}/beds/${bedNumber}?status=${status}`);
      toast.success(`Unit ${bedNumber} redefined as ${status}`);
      fetchBeds();
    } catch (error) {
      toast.error('Sync failed');
    }
  };

  const getStatusConfig = (status) => {
    switch (status) {
      case 'available':
        return { 
          bg: 'bg-emerald-50 text-emerald-600 border-emerald-100 dark:bg-emerald-950/20', 
          icon: BedIcon, 
          label: 'Vacant',
          pulse: false
        };
      case 'occupied':
        return { 
          bg: 'bg-rose-50 text-rose-600 border-rose-100 dark:bg-rose-950/20', 
          icon: Activity, 
          label: 'In-Use',
          pulse: true
        };
      case 'cleaning':
        return { 
          bg: 'bg-amber-50 text-amber-600 border-amber-100 dark:bg-amber-950/20', 
          icon: Wind, 
          label: 'Sanitizing',
          pulse: false
        };
      default:
        return { bg: 'bg-slate-50 border-slate-100', icon: Info, label: 'Unknown' };
    }
  };

  const stats = {
    available: beds.filter(b => b.status === 'available').length,
    occupied: beds.filter(b => b.status === 'occupied').length,
    cleaning: beds.filter(b => b.status === 'cleaning').length
  };

  if (loading) {
    return (
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <Loader2 className="w-10 h-10 text-teal-500 animate-spin mb-4" />
          <p className="text-xs font-black text-slate-400 uppercase tracking-widest text-center">Interfacing with Bed Sensors...</p>
        </div>
      );
  }

  return (
    <div className="max-w-[1600px] mx-auto space-y-6 animate-in fade-in duration-700">
      
      {/* ── Resource Header ── */}
      <div className="bg-white rounded-[2.5rem] p-8 md:p-10 shadow-xl border border-slate-100 flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="space-y-2">
           <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 bg-teal-50 text-teal-600 text-[10px] font-black uppercase rounded border border-teal-100">Live Inventory</span>
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Facility Operations</span>
           </div>
           <h1 className="text-4xl font-black text-slate-800 tracking-tight">Resource <span className="text-teal-600">Allocation</span></h1>
           <p className="text-slate-500 font-medium max-w-lg text-sm leading-relaxed">
             Monitor ward utilization in real-time. Toggle unit status for sanitization or maintenance.
           </p>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
           {[
             { label: 'Vacant', count: stats.available, color: 'emerald' },
             { label: 'Active', count: stats.occupied, color: 'rose' },
             { label: 'Process', count: stats.cleaning, color: 'amber' },
           ].map((s, i) => (
             <div key={i} className="px-6 py-4 bg-slate-50 rounded-3xl border border-slate-100 text-center min-w-[100px]">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">{s.label}</p>
                <p className={`text-2xl font-black text-${s.color}-600`}>{s.count}</p>
             </div>
           ))}
        </div>
      </div>

      {/* ── Floor Map Visualizer ── */}
      <Card className="border-none shadow-2xl rounded-[3rem] overflow-hidden bg-white ring-1 ring-slate-100">
        <CardHeader className="p-10 pb-0 border-b border-slate-50 space-y-4">
           <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                 <div className="w-12 h-12 rounded-2xl bg-teal-500/10 text-teal-600 flex items-center justify-center">
                    <Layout className="w-6 h-6" />
                 </div>
                 <div>
                    <CardTitle className="text-xl font-black text-slate-800">Operational Unit Grid</CardTitle>
                    <p className="text-xs text-slate-400 font-bold uppercase tracking-widest">Interactive facility floorplan control</p>
                 </div>
              </div>
              <div className="flex items-center gap-6">
                 <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                    <span className="text-[10px] font-black opacity-50 uppercase tracking-widest">Available</span>
                 </div>
                 <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-rose-500" />
                    <span className="text-[10px] font-black opacity-50 uppercase tracking-widest">Occupied</span>
                 </div>
                 <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-amber-500" />
                    <span className="text-[10px] font-black opacity-50 uppercase tracking-widest">Sanitizing</span>
                 </div>
              </div>
           </div>
        </CardHeader>
        
        <CardContent className="p-10 pt-8">
           <div className="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-10 gap-4">
              {beds.map((bed) => {
                 const config = getStatusConfig(bed.status);
                 const Icon = config.icon;
                 return (
                    <div key={bed.bed_number} className="relative group">
                       <button
                         data-testid={`bed-${bed.bed_number}`}
                         onClick={() => {
                           if (bed.status === 'available') updateBedStatus(bed.bed_number, 'cleaning');
                           else if (bed.status === 'cleaning') updateBedStatus(bed.bed_number, 'available');
                         }}
                         disabled={bed.status === 'occupied'}
                         className={`w-full aspect-square md:aspect-auto md:h-28 rounded-3xl border-2 flex flex-col items-center justify-center gap-2 transition-all duration-300 relative overflow-hidden ${config.bg} ${
                           bed.status === 'occupied' 
                             ? 'cursor-not-allowed border-rose-100' 
                             : 'hover:scale-105 hover:shadow-xl hover:border-teal-400 cursor-pointer border-transparent'
                         }`}
                       >
                         {config.pulse && (
                            <div className="absolute top-2 right-2 flex h-2 w-2">
                               <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                               <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-600"></span>
                            </div>
                         )}
                         <Icon className={`w-6 h-6 ${config.pulse ? 'animate-pulse' : ''}`} />
                         <span className="text-xs font-black tracking-tight">#{bed.bed_number}</span>
                         <span className="text-[8px] font-black uppercase tracking-widest opacity-60 leading-none">{config.label}</span>
                       </button>
                    </div>
                 );
              })}
           </div>

           <div className="mt-12 p-8 bg-slate-900 rounded-[2.5rem] flex items-center justify-between text-white relative overflow-hidden group">
              <div className="absolute right-[-10%] top-[-50%] w-64 h-64 bg-teal-500/10 rounded-full blur-3xl" />
              <div className="flex items-center gap-6 relative z-10">
                 <div className="w-14 h-14 rounded-3xl bg-white/10 flex items-center justify-center text-teal-400">
                    <UserCheck className="w-7 h-7" />
                 </div>
                 <div>
                    <h4 className="font-black text-lg leading-tight">Smart Queue Integration</h4>
                    <p className="text-xs font-medium text-slate-400">The AI model prefers Vacant beds for trauma admissions.</p>
                 </div>
              </div>
              <div className="hidden lg:flex items-center gap-3 relative z-10">
                 <Badge className="bg-white/10 text-[10px] font-black px-4 py-2 hover:bg-white/20 border-none transition-colors">AUTO-ASSIGN ENABLED</Badge>
              </div>
           </div>
        </CardContent>
      </Card>

    </div>
  );
};

export default BedManagement;
