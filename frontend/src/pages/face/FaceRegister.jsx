import { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle, XCircle } from 'lucide-react';
import api from '../../services/api';

export default function FaceRegister() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [status, setStatus] = useState('initializing');
  const [message, setMessage] = useState('Preparing AI face models...');
  const [modelReady, setModelReady] = useState(false);
  const [stream, setStream] = useState(null);
  const [faceCount, setFaceCount] = useState(0);
  const [role, setRole] = useState('student');
  const [students, setStudents] = useState([]);
  const [selectedStudent, setSelectedStudent] = useState('');

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const { data: session } = await api.get('/api/session', { noCache: true });
        const userRole = session.user?.role || 'student';
        setRole(userRole);
        if (userRole === 'admin') {
          const { data } = await api.get('/api/face/students', { noCache: true });
          setStudents(data.students || []);
          setSelectedStudent(data.students?.[0]?.student_id || '');
        } else {
          const { data } = await api.get('/api/face/students/me', { noCache: true });
          setFaceCount(data.embedding_count || 0);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchInfo();
    api.get('/api/face/status')
      .then(() => {
        setModelReady(true);
        setStatus(current => current === 'initializing' ? 'ready' : current);
        setMessage(current => current === 'Preparing AI face models...'
          ? 'Position your face in the frame and click capture.'
          : current);
      })
      .catch((error) => {
        setStatus('error');
        setMessage(error.response?.data?.message || 'Face recognition models are unavailable.');
      });
    startCamera();
    return () => streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function startCamera() {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      if (!videoRef.current) {
        mediaStream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
    } catch (err) {
      console.error('Error accessing camera', err);
      setStatus('error');
      setMessage('Camera access denied or unavailable.');
    }
  }

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
  };

  const handleCapture = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      setStatus('error');
      setMessage('Camera is still starting. Please try again.');
      return;
    }

    setStatus('scanning');
    setMessage('Processing face embedding...');

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL('image/jpeg', 0.82);

    try {
      const response = await api.post('/api/face/register', {
        image_b64: base64Image,
        ...(role === 'admin' ? { student_id: selectedStudent } : {})
      });

      const data = response.data;

      if (data.success) {
        setStatus('success');
        setMessage(data.message || 'Face registered successfully.');
        setFaceCount(data.embedding_count ?? faceCount + 1);
        setStudents((current) => current.map((student) => student.student_id === selectedStudent ? { ...student, face_registered: true, embedding_count: (student.embedding_count || 0) + 1 } : student));
        setTimeout(() => {
          setStatus('ready');
          setMessage('Capture another angle for better accuracy.');
        }, 2000);
      } else {
        setStatus('error');
        setMessage(data.error || 'Failed to register face.');
        setTimeout(() => {
          setStatus('ready');
          setMessage('Try again.');
        }, 3000);
      }
    } catch (error) {
      setStatus('error');
      setMessage(error.response?.data?.error || 'Server error.');
      setTimeout(() => setStatus('ready'), 3000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">{role === 'admin' ? 'Student Face Registration' : 'Biometric Registration'}</h1>
        <p className="text-secondary">Register your face for AI-powered attendance and wellness checks.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          {role === 'admin' ? <label className="block space-y-2 text-sm text-secondary">Student<select value={selectedStudent} onChange={(event) => setSelectedStudent(event.target.value)} className="glass-input" required><option value="">Select student</option>{students.map((student) => <option key={student.student_id} value={student.student_id}>{student.name} · {student.roll_no}</option>)}</select></label> : null}
          <div className="glass-card overflow-hidden relative aspect-video bg-black/50 flex items-center justify-center">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="w-full h-full object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />
            
            {stream && (
              <div className="absolute inset-0 border-2 border-white/20 pointer-events-none flex items-center justify-center">
                 <div className="w-48 h-64 border-2 border-dashed border-cyan-500/50 rounded-full"></div>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <button type="button" onClick={stream ? stopCamera : startCamera} disabled={status === 'scanning'} className="glass-button flex-1 py-3">{stream ? 'Stop camera' : 'Start camera'}</button>
            <button onClick={handleCapture} disabled={!modelReady || status === 'scanning' || !stream || (role === 'admin' && !selectedStudent)} className="glass-button flex-1 bg-white py-3 text-black hover:bg-white/90 disabled:opacity-50">
              {status === 'scanning' ? 'Processing...' : 'Capture face'}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="font-semibold text-lg mb-4">Registration Status</h3>
            {role === 'admin' ? <div className="text-4xl font-bold mb-2">{students.find((student) => student.student_id === selectedStudent)?.embedding_count || 0} <span className="text-lg text-secondary font-normal">Embeddings Saved</span></div> : <div className="text-4xl font-bold mb-2">{faceCount} <span className="text-lg text-secondary font-normal">Embeddings Saved</span></div>}
            <p className="text-sm text-secondary">
              We recommend capturing at least 3-5 different angles (front, slight left, slight right, looking up/down) for optimal recognition accuracy.
            </p>
          </div>

          <div className={`glass-card p-6 transition-colors ${
              status === 'success' ? 'bg-green-500/10 border-green-500/30' :
              status === 'error' ? 'bg-red-500/10 border-red-500/30' :
              ''
            }`}>
            <div className="flex items-center gap-3">
              {status === 'success' && <CheckCircle className="w-6 h-6 text-green-400" />}
              {status === 'error' && <XCircle className="w-6 h-6 text-red-400" />}
              {status === 'scanning' && <Camera className="w-6 h-6 text-cyan-400 animate-pulse" />}
              <span className="font-medium">{message}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
