import { Link, useLocation } from 'react-router-dom';
import { Activity, Bed, ClipboardList, BarChart3, Stethoscope, LogOut, ShieldCheck, User } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Layout = ({ children, user, onLogout }) => {
  const location = useLocation();

  const allNavItems = [
    { path: '/', label: 'Dashboard', icon: Activity },
    { path: '/beds', label: 'Bed Management', icon: Bed },
    { path: '/predict', label: 'Predict Stay', icon: Stethoscope },
    { path: '/patients', label: 'Patient Records', icon: ClipboardList },
    { path: '/analytics', label: 'Analytics', icon: BarChart3 }
  ];

  // Filter nav items based on user permissions
  const navItems = allNavItems.filter(item => 
    user?.permissions?.includes(item.label) || item.label === 'Dashboard'
  );

  const roleColors = {
    Admin: 'bg-emerald-500 text-white shadow-emerald-500/20',
    Doctor: 'bg-indigo-500 text-white shadow-indigo-500/20',
    Nurse: 'bg-purple-500 text-white shadow-purple-500/20',
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans selection:bg-teal-500/20">
      
      {/* ── Premium Top Navigation Bar ── */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-100 sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 h-20 flex items-center justify-between">
          
          <div className="flex items-center gap-10">
            {/* Brand Logo */}
            <div className="flex items-center gap-3 group transition-all duration-300">
              <div className="w-11 h-11 bg-slate-900 rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform duration-500">
                <Activity className="w-6 h-6 text-teal-400" />
              </div>
              <div>
                <h1 className="text-xl font-black text-slate-800 tracking-tight leading-none mb-1">HEALTHCARE <span className="text-teal-600">OS</span></h1>
                <p className="text-[10px] text-slate-400 font-black uppercase tracking-[0.2em]">Clinical Decision Suite</p>
              </div>
            </div>

            {/* Main Nav Links */}
            <nav className="hidden xl:flex items-center gap-1 px-1 py-1 rounded-2xl bg-slate-100/50">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-2 px-4 py-2 text-[11px] font-black uppercase tracking-wider rounded-xl transition-all duration-300 ${
                      isActive
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-teal-500' : ''}`} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* User Profile & Actions */}
          {user && (
            <div className="flex items-center gap-6 pl-6 border-l border-slate-100">
               <div className="flex items-center gap-4">
                  <div className="text-right">
                     <p className="text-xs font-black text-slate-800 leading-none mb-1">{user.name}</p>
                     <div className="flex items-center justify-end">
                        <span className={`px-2 py-0.5 rounded-full text-[8px] font-black uppercase tracking-[0.1em] shadow-sm ${roleColors[user.role]}`}>
                           {user.role}
                        </span>
                     </div>
                  </div>
                  <div className="w-10 h-10 rounded-full bg-slate-200 p-0.5 border border-slate-100 shadow-inner group cursor-pointer overflow-hidden">
                     <div className="w-full h-full rounded-full bg-white flex items-center justify-center">
                        <User className="w-5 h-5 text-slate-300 group-hover:text-slate-600 transition-colors" />
                     </div>
                  </div>
               </div>

               <Button
                 onClick={onLogout}
                 className="h-10 w-10 p-0 rounded-xl bg-rose-50 hover:bg-rose-500 text-rose-600 hover:text-white border border-rose-100 hover:border-rose-500 shadow-sm transition-all active:scale-95 group"
                 title="Secure Logout"
               >
                 <LogOut className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
               </Button>
            </div>
          )}
        </div>
      </header>

      {/* ── Sub-Nav for Mobile/Small Screens (Optional but nice) ── */}
      <nav className="xl:hidden bg-white border-b border-slate-100 sticky top-20 z-40 overflow-x-auto no-scrollbar">
        <div className="flex px-4 py-2">
           {navItems.map((item) => {
             const Icon = item.icon;
             const isActive = location.pathname === item.path;
             return (
               <Link
                 key={item.path}
                 to={item.path}
                 className={`flex-shrink-0 flex items-center gap-2 px-4 py-3 text-[10px] font-black uppercase tracking-widest border-b-2 transition-all ${
                   isActive ? 'text-teal-600 border-teal-600' : 'text-slate-400 border-transparent'
                 }`}
               >
                 <Icon className="w-3.5 h-3.5" />
                 {item.label}
               </Link>
             );
           })}
        </div>
      </nav>

      {/* ── Main Viewport Content ── */}
      <main className="max-w-[1600px] mx-auto p-6 md:p-10 pb-20">
        {children}
      </main>

      {/* ── Floating Systems Footer ── */}
      <footer className="fixed bottom-6 left-1/2 -translate-x-1/2 h-10 px-6 bg-slate-900 rounded-full flex items-center gap-4 text-white shadow-2xl shadow-slate-900/40 z-50 border border-slate-800">
         <div className="flex items-center gap-2 px-2 border-r border-slate-700 h-1/2">
            <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
            <span className="text-[9px] font-black opacity-80 uppercase tracking-widest">Systems Nominal</span>
         </div>
         <div className="flex items-center gap-2">
            <ShieldCheck className="w-3 h-3 text-teal-400" />
            <span className="text-[9px] font-black opacity-50 uppercase tracking-[0.2em]">Encryption Active: v5.2</span>
         </div>
      </footer>

    </div>
  );
};

export default Layout;
