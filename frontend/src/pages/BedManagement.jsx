import { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Bed as BedIcon } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const BedManagement = () => {
  const [beds, setBeds] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBeds();
    const interval = setInterval(fetchBeds, 5000); // Refresh every 5 seconds
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
      toast.success(`Bed ${bedNumber} status updated to ${status}`);
      fetchBeds();
    } catch (error) {
      toast.error('Failed to update bed status');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'available':
        return 'bg-emerald-100 border-emerald-300 text-emerald-700 hover:bg-emerald-200';
      case 'occupied':
        return 'bg-red-100 border-red-300 text-red-700 hover:bg-red-200';
      case 'cleaning':
        return 'bg-amber-100 border-amber-300 text-amber-700 hover:bg-amber-200';
      default:
        return 'bg-slate-100 border-slate-300';
    }
  };

  const stats = {
    available: beds.filter(b => b.status === 'available').length,
    occupied: beds.filter(b => b.status === 'occupied').length,
    cleaning: beds.filter(b => b.status === 'cleaning').length
  };

  if (loading) {
    return <div className="text-center py-8">Loading beds...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-900 tracking-tight">Bed Management</h1>
        <p className="text-base text-slate-600 mt-1">Real-time bed status and availability</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600">Available</p>
                <p className="text-3xl font-bold font-mono text-slate-900">{stats.available}</p>
              </div>
              <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center">
                <BedIcon className="w-6 h-6 text-emerald-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-red-500">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600">Occupied</p>
                <p className="text-3xl font-bold font-mono text-slate-900">{stats.occupied}</p>
              </div>
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <BedIcon className="w-6 h-6 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600">Cleaning</p>
                <p className="text-3xl font-bold font-mono text-slate-900">{stats.cleaning}</p>
              </div>
              <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center">
                <BedIcon className="w-6 h-6 text-amber-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bed Grid */}
      <Card>
        <CardContent className="p-6">
          <div className="mb-4 flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-emerald-500 rounded"></div>
              <span>Available</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>Occupied</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-amber-500 rounded"></div>
              <span>Cleaning</span>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-6 lg:grid-cols-10 gap-3">
            {beds.map((bed) => (
              <button
                key={bed.bed_number}
                data-testid={`bed-${bed.bed_number}`}
                onClick={() => {
                  if (bed.status === 'available') {
                    updateBedStatus(bed.bed_number, 'cleaning');
                  } else if (bed.status === 'cleaning') {
                    updateBedStatus(bed.bed_number, 'available');
                  }
                }}
                disabled={bed.status === 'occupied'}
                className={`p-4 rounded-lg border-2 flex flex-col items-center justify-center space-y-1 transition-colors ${
                  getStatusColor(bed.status)
                } ${bed.status === 'occupied' ? 'cursor-not-allowed opacity-70' : 'cursor-pointer'}`}
              >
                <BedIcon className="w-5 h-5" />
                <span className="text-sm font-bold font-mono">{bed.bed_number}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default BedManagement;
