import { Link, useLocation } from 'react-router-dom';
import { Activity, Bed, ClipboardList, BarChart3, Stethoscope } from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Activity },
    { path: '/beds', label: 'Bed Management', icon: Bed },
    { path: '/predict', label: 'Predict Stay', icon: Stethoscope },
    { path: '/patients', label: 'Patient Records', icon: ClipboardList },
    { path: '/analytics', label: 'Analytics', icon: BarChart3 }
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="w-full mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-teal-600 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Hospital BMS</h1>
                <p className="text-xs text-slate-500 font-mono">ML Capstone Project</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200">
        <div className="w-full mx-auto px-6">
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  data-testid={`nav-${item.label.toLowerCase().replace(' ', '-')}`}
                  className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 hover:text-teal-600 hover:border-teal-600 ${
                    isActive
                      ? 'text-teal-600 border-teal-600'
                      : 'text-slate-600 border-transparent'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="w-full mx-auto px-6 py-8">
        {children}
      </main>
    </div>
  );
};

export default Layout;
