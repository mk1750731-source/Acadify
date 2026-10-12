import { useEffect, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, BookOpen, Clock, Calendar, CheckSquare, Bell, User, LogOut, FileText, Menu, ShieldAlert, Camera, ScanFace, History, Medal, Calculator, Library, Settings, Award, Activity, Users, BarChart3, ClipboardList, Building2, MessageSquare, Database, ScrollText, UserCog, HandHeart } from 'lucide-react';
import api from '../services/api';

const Logo = () => (
  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-green-600 shadow-[0_2px_8px_rgba(22,163,74,0.35)] shrink-0">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M12 2L2 7l10 5 10-5-10-5z" fill="white" />
      <path d="M2 17l10 5 10-5" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2 12l10 5 10-5" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </div>
);

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [role, setRole] = useState('student');
  const location = useLocation();

  useEffect(() => {
    const cached = localStorage.getItem('edu_user');
    if (cached) {
      try { setRole(JSON.parse(cached).role || 'student'); } catch {}
      return;
    }
    api.get('/api/session', { noCache: true })
      .then(({ data }) => {
        if (data.user) {
          setRole(data.user.role || 'student');
          localStorage.setItem('edu_user', JSON.stringify(data.user));
        } else {
          localStorage.removeItem('edu_user');
          window.location.href = '/login';
        }
      })
      .catch(() => {
        localStorage.removeItem('edu_user');
        window.location.href = '/login';
      });
  }, []);

  const handleLogout = async () => {
    localStorage.removeItem('edu_user');
    try { await api.get('/logout'); } catch {}
    window.location.href = '/login';
  };

  const studentMenuItems = [
    { name: 'Dashboard', icon: LayoutDashboard, path: '/student/dashboard' },
    { name: 'Academics', icon: BookOpen, path: '/student/academics' },
    { name: 'Attendance', icon: CheckSquare, path: '/student/attendance' },
    { name: 'Marks & Performance', icon: BarChart3, path: '/student/marks' },
    { name: 'Timetable', icon: Clock, path: '/student/timetable' },
    { name: 'Assignments', icon: FileText, path: '/student/assignments' },
    { name: 'Exams', icon: Calendar, path: '/student/exams' },
    { name: 'Counseling', icon: ShieldAlert, path: '/student/counseling' },
    { name: 'Profile', icon: User, path: '/student/profile' },
    { name: 'AI Predictions', icon: Activity, path: '/student/predictions' },
    { name: 'Notifications', icon: Bell, path: '/student/notifications' },
    { name: 'Leaderboard', icon: Medal, path: '/student/leaderboard' },
    { name: 'GPA Simulator', icon: Calculator, path: '/student/gpa-simulator' },
    { name: 'Study Resources', icon: Library, path: '/student/resources' },
    { name: 'Quick Notes', icon: FileText, path: '/student/notes' },
    { name: 'Campus Pass', icon: Award, path: '/student/pass' },
    { name: 'Settings', icon: Settings, path: '/student/settings' },
    { name: 'Register Face', icon: ScanFace, path: '/face/register' },
    { name: 'Face History', icon: History, path: '/face/history' },
  ];

  const adminMenuItems = [
    { name: 'Command Center', icon: LayoutDashboard, path: '/admin/dashboard' },
    { name: 'Student Directory', icon: Users, path: '/admin/students' },
    { name: 'Risk Predictions', icon: Activity, path: '/admin/predictions' },
    { name: 'Run Prediction', icon: ShieldAlert, path: '/admin/predict' },
    { name: 'Attendance Control', icon: ClipboardList, path: '/admin/attendance' },
    { name: 'Reports', icon: BarChart3, path: '/admin/reports' },
    { name: 'Counseling Requests', icon: MessageSquare, path: '/admin/counseling' },
    { name: 'Interventions', icon: HandHeart, path: '/admin/interventions' },
    { name: 'Departments', icon: Building2, path: '/admin/departments' },
    { name: 'Analytics', icon: BarChart3, path: '/admin/analytics' },
    { name: 'CSV Import', icon: Database, path: '/admin/data-upload' },
    { name: 'Bulk Notifications', icon: Bell, path: '/admin/notifications' },
    { name: 'User Accounts', icon: UserCog, path: '/admin/users' },
    { name: 'Audit Stream', icon: ScrollText, path: '/admin/audit-logs' },
    { name: 'System Settings', icon: Settings, path: '/admin/settings' },
    { name: 'AI Face Attendance', icon: Camera, path: '/face/attendance' },
    { name: 'Register Student Face', icon: ScanFace, path: '/face/register' },
    { name: 'Face History', icon: History, path: '/face/history' },
  ];

  const teacherMenuItems = [
    { name: 'Dashboard', icon: LayoutDashboard, path: '/teacher/dashboard' },
    { name: 'Attendance', icon: CheckSquare, path: '/teacher/attendance' },
    { name: 'Marks', icon: BarChart3, path: '/teacher/marks' },
    { name: 'Timetable', icon: Clock, path: '/teacher/timetable' },
    { name: 'Exams', icon: Calendar, path: '/teacher/exams' },
    { name: 'Assignments', icon: FileText, path: '/teacher/assignments' },
    { name: 'Verify Exceptions', icon: CheckSquare, path: '/teacher/verify' },
    { name: 'AI Face Attendance', icon: Camera, path: '/face/attendance' },
    { name: 'Face History', icon: History, path: '/face/history' },
  ];

  const parentMenuItems = [
    { name: 'Dashboard', icon: Users, path: '/parent/dashboard' },
  ];

  const menuItems = role === 'admin' ? adminMenuItems : role === 'teacher' ? teacherMenuItems : role === 'parent' ? parentMenuItems : studentMenuItems;

  const roleLabel = role === 'admin' ? 'Admin' : role === 'teacher' ? 'Teacher' : role === 'parent' ? 'Family' : 'Student';

  return (
    <div className="min-h-screen flex overflow-hidden" style={{ background: '#f0f7f1' }}>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed lg:static inset-y-0 left-0 z-50 w-64 flex flex-col
        bg-white border-r border-green-100 shadow-[2px_0_16px_rgba(20,83,45,0.06)]
        transform transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Brand */}
        <div className="h-16 flex items-center gap-2.5 px-5 border-b border-green-100">
          <Logo />
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-green-600">Acadify</p>
            <p className="text-xs text-gray-400 leading-none">{roleLabel} Portal</p>
          </div>
        </div>

        {/* Menu */}
        <div className="flex-1 overflow-y-auto py-4 px-3 space-y-0.5">
          {menuItems.map((item) => {
            const isActive = location.pathname.startsWith(item.path);
            return (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-600 text-white shadow-[0_2px_8px_rgba(22,163,74,0.35)]'
                    : 'text-gray-700 hover:bg-green-50 hover:text-emerald-700'
                }`}
              >
                <item.icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-white' : 'text-gray-500'}`} />
                <span>{item.name}</span>
                {isActive && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-white/70" />}
              </Link>
            );
          })}
        </div>

        {/* Logout */}
        <div className="p-3 border-t border-green-100">
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm font-medium text-red-500 hover:bg-red-50 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">

        {/* Top bar */}
        <header className="h-16 bg-white border-b border-green-100 shadow-[0_1px_8px_rgba(20,83,45,0.05)] flex items-center justify-between px-4 lg:px-8 z-30 shrink-0">
          <button className="lg:hidden p-2 text-gray-500 hover:text-gray-800" onClick={() => setSidebarOpen(true)}>
            <Menu className="w-5 h-5" />
          </button>

          <p className="hidden lg:block text-sm font-medium text-gray-500">
            {menuItems.find(i => location.pathname.startsWith(i.path))?.name || 'Portal'}
          </p>

          <div className="flex items-center gap-3">
            <button className="relative p-2 text-gray-400 hover:text-green-600 transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-green-500 rounded-full" />
            </button>
            <div className="w-8 h-8 rounded-full bg-green-100 border border-green-200 flex items-center justify-center">
              <User className="w-4 h-4 text-green-600" />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
