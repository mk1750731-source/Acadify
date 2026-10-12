import { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, CheckCircle, XCircle, AlertCircle, Users, RefreshCw, Plus, Calendar, ChevronRight, Loader } from 'lucide-react';
import api from '../../services/api';
import { loadModels, getEmbedding } from '../../services/faceApi';

export default function FaceAttendance() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const autoScanRef = useRef(null);
  const scanningRef = useRef(false);

  const [step, setStep] = useState(1);

  // Step 1
  const [subjects, setSubjects] = useState([]);
  const [subjectsLoading, setSubjectsLoading] = useState(true);
  const [form, setForm] = useState({ subject: '', date: new Date().toISOString().slice(0, 10), label: '' });
  const [creating, setCreating] = useState(false);
  const [todaySessions, setTodaySessions] = useState([]);
  const [initError, setInitError] = useState('');

  // Step 2
  const [session, setSession] = useState(null);
  const [classStudents, setClassStudents] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [stream, setStream] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Load subjects
    api.get('/api/face/subjects', { noCache: true })
      .then(r => {
        setSubjects(r.data.subjects || []);
        setSubjectsLoading(false);
      })
      .catch(() => setSubjectsLoading(false));

    // Pre-load face models in background (don't block UI)
    loadModels()
      .then(() => setModelsLoaded(true))
      .catch(() => {}); // will retry on first scan

    loadTodaySessions();

    return () => {
      streamRef.current?.getTracks().forEach(t => t.stop());
      clearTimeout(autoScanRef.current);
    };
  }, []);

  const loadTodaySessions = () => {
    api.get('/api/face/sessions', { noCache: true })
      .then(r => setTodaySessions(r.data.sessions || []))
      .catch(() => {});
  };

  const handleCreateSession = async () => {
    if (!form.subject) return;
    setCreating(true);
    setInitError('');
    try {
      const { data } = await api.post('/api/face/session/create', form);
      if (data.success) {
        await enterSession(data);
        loadTodaySessions();
      } else {
        setInitError(data.message || 'Failed to create session.');
      }
    } catch (e) {
      setInitError(e.response?.data?.message || 'Failed to create session.');
    } finally {
      setCreating(false);
    }
  };

  const enterSession = async (sess) => {
    setSession(sess);
    setStep(2);
    setScanResult(null);
    setMessage('');
    try {
      const { data } = await api.get(
        `/api/face/class-students?subject=${encodeURIComponent(sess.subject)}&session_id=${sess.session_id}`,
        { noCache: true }
      );
      const students = data.students || [];
      setClassStudents(students);
      const firstPending = students.findIndex(s => !s.already_marked);
      setCurrentIdx(firstPending === -1 ? 0 : firstPending);
    } catch {
      setMessage('Failed to load student list.');
    }
    await startCamera();
  };

  const startCamera = async () => {
    try {
      const ms = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      streamRef.current = ms;
      if (videoRef.current) videoRef.current.srcObject = ms;
      setStream(true);
    } catch {
      setMessage('Camera access denied. Please allow camera permissions.');
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setStream(false);
  };

  const classStudentsRef = useRef(classStudents);
  const currentIdxRef = useRef(currentIdx);
  useEffect(() => { classStudentsRef.current = classStudents; }, [classStudents]);
  useEffect(() => { currentIdxRef.current = currentIdx; }, [currentIdx]);

  const advanceToNext = useCallback((students, afterIdx) => {
    const next = students.findIndex((s, i) => i > afterIdx && !s.already_marked);
    if (next !== -1) {
      setCurrentIdx(next);
      setMessage(`Ready for: ${students[next].name}`);
      setScanResult(null);
    } else {
      setMessage('✓ All students have been scanned!');
      setScanResult(null);
    }
  }, []);

  const sessionRef = useRef(session);
  useEffect(() => { sessionRef.current = session; }, [session]);

  const doScan = useCallback(async (sessionArg) => {
    if (scanningRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) {
      setMessage('Camera not ready — wait a moment.');
      return;
    }

    scanningRef.current = true;
    setScanning(true);
    setScanResult(null);

    try {
      // Load models if not yet loaded (first scan)
      if (!modelsLoaded) {
        setMessage('Loading face models...');
        await loadModels();
        setModelsLoaded(true);
      }

      setMessage('Detecting face...');
      const embedding = await getEmbedding(video);
      if (!embedding) {
        setScanResult({ recognized: false });
        setMessage('No face detected — reposition and try again.');
        return;
      }

      setMessage('Scanning...');
      const { data } = await api.post('/api/face/recognize', {
        embedding,
        subject: sessionArg.subject,
        session_id: sessionArg.session_id,
      });

      if (data.success && data.recognized) {
        const result = {
          name: data.student_name,
          roll_no: data.roll_no,
          already_marked: data.already_marked,
          expression: data.expression,
          expression_confidence: data.expression_confidence,
        };
        setScanResult(result);
        setMessage(data.already_marked
          ? `${data.student_name} — already marked today`
          : `✓ ${data.student_name} marked present`
        );
        setClassStudents(prev => {
          const updated = prev.map(s =>
            s.roll_no === data.roll_no ? { ...s, today_status: 'present', already_marked: true } : s
          );
          autoScanRef.current = setTimeout(() => {
            advanceToNext(updated, currentIdxRef.current);
          }, 1500);
          return updated;
        });
      } else {
        setScanResult({ recognized: false });
        setMessage(data.message || data.error || 'Face not recognized — reposition and try again.');
      }
    } catch (err) {
      setScanResult({ recognized: false });
      setMessage(err.response?.data?.error || err.response?.data?.message || 'Scan failed.');
    } finally {
      scanningRef.current = false;
      setScanning(false);
    }
  }, [modelsLoaded, advanceToNext]);

  const handleScan = useCallback(() => {
    if (!sessionRef.current || !stream) return;
    doScan(sessionRef.current);
  }, [stream, doScan]);

  const markedCount = classStudents.filter(s => s.already_marked || s.today_status === 'present').length;
  const allDone = classStudents.length > 0 && markedCount === classStudents.length;

  const exitSession = () => {
    stopCamera();
    clearTimeout(autoScanRef.current);
    scanningRef.current = false;
    setStep(1);
    setSession(null);
    setClassStudents([]);
    setScanResult(null);
    setMessage('');
    setScanning(false);
    loadTodaySessions();
  };

  const refreshStudents = async () => {
    if (!session) return;
    try {
      const { data } = await api.get(
        `/api/face/class-students?subject=${encodeURIComponent(session.subject)}&session_id=${session.session_id}`,
        { noCache: true }
      );
      setClassStudents(data.students || []);
    } catch {}
  };

  // ── Step 1: Session Setup ──────────────────────────────────────────────────
  if (step === 1) {
    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="glass-card p-6 border-l-4 border-l-cyan-500">
          <h1 className="text-2xl font-bold mb-1">Face Attendance</h1>
          <p className="text-secondary text-sm">Create a session for a subject, then scan student faces.</p>
        </div>

        <div className="glass-card p-6 space-y-4">
          <h2 className="font-semibold flex items-center gap-2"><Plus className="w-4 h-4 text-cyan-400" /> New Session</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-xs text-secondary block mb-1">Subject *</label>
              {subjectsLoading ? (
                <div className="glass-input w-full flex items-center gap-2 text-secondary text-sm">
                  <Loader className="w-3 h-3 animate-spin" /> Loading subjects...
                </div>
              ) : subjects.length === 0 ? (
                <div className="glass-input w-full text-amber-400 text-sm">
                  No subjects found. Ensure students are enrolled.
                </div>
              ) : (
                <select
                  value={form.subject}
                  onChange={e => setForm(f => ({ ...f, subject: e.target.value }))}
                  className="glass-input w-full"
                >
                  <option value="">-- Select Subject --</option>
                  {subjects.map(s => <option key={s.subject} value={s.subject}>{s.subject}</option>)}
                </select>
              )}
            </div>
            <div>
              <label className="text-xs text-secondary block mb-1">Date *</label>
              <input
                type="date"
                value={form.date}
                onChange={e => setForm(f => ({ ...f, date: e.target.value }))}
                className="glass-input w-full"
              />
            </div>
            <div>
              <label className="text-xs text-secondary block mb-1">Label (optional)</label>
              <input
                type="text"
                placeholder="e.g. Morning, Lab"
                value={form.label}
                onChange={e => setForm(f => ({ ...f, label: e.target.value }))}
                className="glass-input w-full"
              />
            </div>
          </div>
          {initError && <p className="text-sm text-amber-500">{initError}</p>}
          {!modelsLoaded && (
            <p className="text-xs text-secondary flex items-center gap-1">
              <Loader className="w-3 h-3 animate-spin" /> Loading face recognition models in background...
            </p>
          )}
          <button
            onClick={handleCreateSession}
            disabled={!form.subject || creating || subjectsLoading}
            className="glass-button bg-white text-black hover:bg-white/90 px-6 py-2.5 disabled:opacity-40"
          >
            {creating ? 'Creating...' : 'Create Session & Start Scanning'}
          </button>
        </div>

        {todaySessions.length > 0 && (
          <div className="glass-card p-6 space-y-3">
            <h2 className="font-semibold flex items-center gap-2"><Calendar className="w-4 h-4 text-purple-400" /> Today's Sessions</h2>
            {todaySessions.map(sess => (
              <div key={sess.session_id} className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/10">
                <div>
                  <p className="font-medium">{sess.subject} {sess.label && <span className="text-secondary text-sm">· {sess.label}</span>}</p>
                  <p className="text-xs text-secondary">{sess.day}, {sess.date} · {sess.student_count} students · by {sess.created_by_name}</p>
                </div>
                <button onClick={() => enterSession(sess)} className="glass-button px-4 py-2 text-sm flex items-center gap-1">
                  Resume <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // ── Step 2: Face Scanning ──────────────────────────────────────────────────
  const currentStudent = classStudents[currentIdx];

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <div className="glass-card p-4 border-l-4 border-l-cyan-500 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold">{session.subject} {session.label && `· ${session.label}`}</h1>
          <p className="text-secondary text-sm">{session.day}, {session.date} · {markedCount}/{classStudents.length} marked</p>
        </div>
        <div className="flex gap-3 items-center">
          <span className="text-emerald-400 font-semibold text-sm">{markedCount} present</span>
          <span className="text-amber-400 font-semibold text-sm">{classStudents.length - markedCount} pending</span>
          <button onClick={exitSession} className="glass-button px-4 py-2 text-sm text-red-400 border-red-500/30">
            End Session
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Panel */}
        <div className="space-y-4">
          {currentStudent && !allDone && (
            <div className="glass-card p-4 border border-cyan-500/30 bg-cyan-500/5">
              <p className="text-xs text-secondary mb-1">Now scanning</p>
              <p className="font-semibold text-lg">{currentStudent.name}</p>
              <p className="text-sm text-secondary">{currentStudent.roll_no} · {currentStudent.department}</p>
              {!currentStudent.face_registered && (
                <p className="text-xs text-red-400 mt-1">⚠ No face registered — cannot scan this student</p>
              )}
            </div>
          )}

          <div className="glass-card overflow-hidden relative aspect-video bg-black/50 flex items-center justify-center">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
            {stream && (
              <div className={`absolute inset-0 border-2 pointer-events-none transition-colors duration-300 ${
                scanning ? 'border-cyan-500 animate-pulse shadow-[0_0_20px_rgba(6,182,212,0.5)]' :
                scanResult?.recognized === false ? 'border-red-500' :
                scanResult ? 'border-green-500' : 'border-white/20'
              }`} />
            )}
            {!stream && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-secondary">
                <Camera className="w-12 h-12 mb-2 opacity-40" />
                <p className="text-sm">Camera off</p>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <button onClick={stream ? stopCamera : startCamera} className="glass-button flex-1 py-3">
              {stream ? 'Stop Camera' : 'Start Camera'}
            </button>
            <button
              onClick={handleScan}
              disabled={scanning || !stream || allDone}
              className="glass-button flex-1 bg-white py-3 text-black hover:bg-white/90 disabled:opacity-40"
            >
              {scanning ? 'Scanning...' : allDone ? 'All Done ✓' : 'Scan Face'}
            </button>
          </div>

          {(message || scanResult) && (
            <div className={`glass-card p-4 space-y-2 ${
              scanResult?.recognized === false ? 'border-red-500/30 bg-red-500/5' :
              scanResult?.already_marked ? 'border-amber-500/30 bg-amber-500/5' :
              scanResult ? 'border-green-500/30 bg-green-500/5' : ''
            }`}>
              <div className="flex items-center gap-2">
                {scanning && <Loader className="w-5 h-5 text-cyan-400 animate-spin shrink-0" />}
                {!scanning && scanResult?.recognized === false && <XCircle className="w-5 h-5 text-red-400 shrink-0" />}
                {!scanning && scanResult?.already_marked && <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />}
                {!scanning && scanResult && scanResult.recognized !== false && !scanResult.already_marked && <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />}
                <span className="text-sm font-medium">{message}</span>
              </div>
              {scanResult?.name && (
                <div className="grid grid-cols-2 gap-1 text-sm pt-2 border-t border-white/10">
                  <span className="text-secondary">Name</span><span>{scanResult.name}</span>
                  <span className="text-secondary">Roll No</span><span>{scanResult.roll_no}</span>
                  {scanResult.expression && <>
                    <span className="text-secondary">Expression</span>
                    <span className="capitalize">{scanResult.expression} ({scanResult.expression_confidence}%)</span>
                  </>}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Students List */}
        <div className="glass-card p-5 space-y-3 max-h-[600px] flex flex-col">
          <div className="flex items-center gap-2 shrink-0">
            <Users className="w-5 h-5 text-cyan-400" />
            <h2 className="font-semibold">{session.subject} — Students</h2>
            <button onClick={refreshStudents} className="ml-auto text-secondary hover:text-white" title="Refresh">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          {classStudents.length === 0 && (
            <p className="text-sm text-secondary text-center py-4">No students enrolled in this subject.</p>
          )}

          <div className="overflow-y-auto flex-1 space-y-2 pr-1">
            {classStudents.map((student, idx) => {
              const marked = student.already_marked || student.today_status === 'present';
              const isNext = idx === currentIdx && !marked;
              return (
                <div
                  key={student._id}
                  onClick={() => { if (!marked) { setCurrentIdx(idx); setScanResult(null); setMessage(`Ready for: ${student.name}`); } }}
                  className={`flex items-center justify-between gap-3 px-4 py-3 rounded-xl transition-all ${
                    isNext ? 'bg-cyan-500/15 border border-cyan-500/40 cursor-pointer' :
                    marked ? 'bg-emerald-500/10 border border-emerald-500/20' :
                    'bg-white/5 border border-white/5 hover:bg-white/10 cursor-pointer'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                      marked ? 'bg-emerald-500/20 text-emerald-400' :
                      isNext ? 'bg-cyan-500/20 text-cyan-400' :
                      'bg-white/10 text-secondary'
                    }`}>
                      {idx + 1}
                    </div>
                    <div>
                      <p className="text-sm font-medium">{student.name}</p>
                      <p className="text-xs text-secondary">{student.roll_no} · {student.department}</p>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    {marked ? (
                      <span className="text-xs text-emerald-400 flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" /> Present</span>
                    ) : !student.face_registered ? (
                      <span className="text-xs text-red-400">No face</span>
                    ) : isNext ? (
                      <span className="text-xs text-cyan-400">→ Scanning</span>
                    ) : (
                      <span className="text-xs text-secondary">Pending</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {allDone && (
            <div className="shrink-0 text-center py-3 text-emerald-400 font-semibold text-sm border-t border-white/10">
              ✓ All students marked for {session.subject}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
