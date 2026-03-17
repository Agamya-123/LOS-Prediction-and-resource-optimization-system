import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';
import { Activity, Lock, User, ShieldCheck, Zap, Globe } from 'lucide-react';

// Hardcoded users for demo
const USERS = {
  'admin': { password: 'admin123', role: 'Admin', name: 'Dr. Saranya P' },
  'doctor': { password: 'doctor123', role: 'Doctor', name: 'Dr. Agamya Rathour' },
  'nurse': { password: 'nurse123', role: 'Nurse', name: 'Nurse Sufiyan Khan' },
};

const ROLE_PERMISSIONS = {
  Admin: ['Dashboard', 'Bed Management', 'Predict Stay', 'Patient Records', 'Analytics', 'Train Model', 'AI Assistant'],
  Doctor: ['Dashboard', 'Predict Stay', 'Patient Records', 'Analytics', 'AI Assistant'],
  Nurse: ['Dashboard', 'Bed Management', 'Patient Records'],
};

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    setTimeout(() => {
      const user = USERS[username.toLowerCase()];
      if (user && user.password === password) {
        toast.success(`Access Granted: ${user.name}`);
        onLogin({
          username: username.toLowerCase(),
          name: user.name,
          role: user.role,
          permissions: ROLE_PERMISSIONS[user.role],
        });
      } else {
        toast.error('Authentication Error: Token Mismatch');
      }
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 relative overflow-hidden">
      
      {/* ── Background Aesthetics ── */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-teal-500/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-500/5 rounded-full blur-[120px]" />
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'radial-gradient(#1e293b 0.5px, transparent 0.5px)', backgroundSize: '24px 24px' }} />
      </div>
      
      <div className="w-full max-w-[1000px] grid grid-cols-1 md:grid-cols-2 bg-white rounded-[3rem] shadow-[0_32px_64px_-16px_rgba(0,0,0,0.1)] overflow-hidden border border-slate-100 relative z-10 mx-4">
         
         {/* Left Side: Brand & Social Proof */}
         <div className="bg-slate-900 p-12 text-white flex flex-col justify-between relative overflow-hidden">
            <div className="absolute right-[-20%] bottom-[-10%] opacity-10">
               <Globe className="w-64 h-64" />
            </div>
            
            <div className="relative z-10">
               <div className="w-14 h-14 bg-teal-500 rounded-2xl flex items-center justify-center mb-8 shadow-lg shadow-teal-500/20">
                  <Activity className="w-8 h-8 text-white" />
               </div>
               <h1 className="text-4xl font-black tracking-tight leading-tight mb-4">
                  Clinical <br/> <span className="text-teal-400">Decision</span> <br/> Support System
               </h1>
               <p className="text-slate-400 font-medium leading-relaxed max-w-xs">
                  Secured biometric-ready portal for healthcare professionals to manage LOS predictions and ward logistics.
               </p>
            </div>

            <div className="relative z-10 flex items-center gap-4 pt-12 border-t border-white/10">
               <div className="flex -space-x-2">
                  {[1,2,3].map(i => <div key={i} className="w-8 h-8 rounded-full border-2 border-slate-900 bg-slate-700" />)}
               </div>
               <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                  Used by 4,000+ Personnel
               </p>
            </div>
         </div>

         {/* Right Side: Auth Form */}
         <div className="p-12 flex flex-col justify-center">
            <div className="mb-10 text-center md:text-left">
               <h2 className="text-2xl font-black text-slate-800 tracking-tight mb-2">System Login</h2>
               <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Authorized Access Only</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
               <div className="space-y-2">
                  <Label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Identity Identifier</Label>
                  <div className="relative">
                     <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                     <Input
                        type="text"
                        placeholder="e.g. admin"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="h-14 pl-12 rounded-2xl border-slate-100 bg-slate-50/50 focus:bg-white focus:ring-teal-500/20 transition-all font-medium text-slate-700"
                        required
                     />
                  </div>
               </div>

               <div className="space-y-2">
                  <Label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Secure Passphrase</Label>
                  <div className="relative">
                     <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                     <Input
                        type="password"
                        placeholder="••••••••"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="h-14 pl-12 rounded-2xl border-slate-100 bg-slate-50/50 focus:bg-white focus:ring-teal-500/20 transition-all font-medium"
                        required
                     />
                  </div>
               </div>

               <Button
                  type="submit"
                  className="w-full h-14 rounded-2xl bg-slate-900 hover:bg-slate-800 text-white font-black shadow-xl transition-all active:scale-95 text-sm uppercase tracking-widest"
                  disabled={loading}
               >
                  {loading ? (
                     <span className="flex items-center">
                        <Zap className="w-4 h-4 mr-2 animate-pulse text-teal-400" />
                        VALIDATING...
                     </span>
                  ) : 'ESTABLISH LINK'}
               </Button>
            </form>

            <div className="mt-12 pt-8 border-t border-slate-50">
               <div className="flex items-center gap-2 mb-4">
                  <ShieldCheck className="w-4 h-4 text-emerald-500" />
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Demo Credentials</span>
               </div>
               <div className="grid grid-cols-3 gap-2 text-[8px] font-black uppercase tracking-tight">
                  <div className="p-2 bg-slate-50 rounded-xl border border-slate-100">
                     <p className="text-slate-400 mb-0.5">ADMIN</p>
                     <p className="text-slate-700">admin / 123</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-xl border border-slate-100">
                     <p className="text-slate-400 mb-0.5">DOCTOR</p>
                     <p className="text-slate-700">doctor / 123</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-xl border border-slate-100">
                     <p className="text-slate-400 mb-0.5">NURSE</p>
                     <p className="text-slate-700">nurse / 123</p>
                  </div>
               </div>
            </div>
         </div>

      </div>
    </div>
  );
};

export default Login;
export { ROLE_PERMISSIONS };
