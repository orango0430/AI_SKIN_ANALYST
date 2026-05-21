import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ScoreVisualization from '../components/ScoreVisualization';
import './Analysis.css';

export default function Analysis() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();
  const { authFetch } = useAuth();
  const navigate = useNavigate();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return;
    setFile(f);
    setResult(null);
    setError('');
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const startAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const fd = new FormData();
      fd.append('image', file);
      const res = await authFetch('/diagnosis/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '분석에 실패했습니다.');
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analysis-page">
      {loading && (
        <div className="loading-overlay">
          <div className="loading-box">
            <div className="spinner" />
            <h3>AI가 분석 중입니다...</h3>
            <p>피부 상태를 정밀하게 스캔하고 있어요.<br/>잠시만 기다려주세요.</p>
          </div>
        </div>
      )}

      <div className="analysis-layout">
        {/* 업로드 */}
        <div className="panel">
          <div className="panel-title"><span>📷</span> 피부 사진 업로드</div>
          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => handleFile(e.target.files[0])} />
            <div className="upload-icon">📁</div>
            <h4>사진을 끌어놓거나 클릭하여 선택</h4>
            <p>얼굴 정면 사진을 업로드해 주세요<br/>자연광 환경에서 찍은 사진이 좋아요</p>
            <div className="formats">
              {['JPG','PNG','WEBP','최대 10MB'].map(f => <span key={f} className="format-badge">{f}</span>)}
            </div>
          </div>

          {preview && <img src={preview} alt="미리보기" className="preview-img" />}

          <div className="tip-box">💡 <strong>촬영 팁:</strong> 세안 후 자연광 아래에서, 화장 없이 정면을 바라보고 찍으면 더 정확해요</div>

          {error && <div className="error-box">⚠️ {error}</div>}

          <button className="analyze-btn" disabled={!preview || loading} onClick={startAnalysis}>
            🔍 AI 분석 시작하기
          </button>
        </div>

        {/* 결과 */}
        <div className="panel">
          <div className="panel-title">
            <span>📊</span> 분석 결과
            {result && (
              <button className="mypage-link" onClick={() => navigate('/mypage')}>
                📁 내 기록 보기
              </button>
            )}
          </div>
          {!result ? (
            <div className="result-empty">
              <div className="empty-icon">🧬</div>
              <p>사진을 업로드하고<br/><strong>AI 분석 시작</strong>을 눌러주세요<br/><br/>분석 결과가 여기에 표시됩니다</p>
            </div>
          ) : (
            <ScoreVisualization result={result} />
          )}
        </div>
      </div>
    </div>
  );
}
