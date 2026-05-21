import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ScoreVisualization from '../components/ScoreVisualization';
import './MyPage.css';

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const scoreColor = (v) => v >= 80 ? '#27ae60' : v >= 60 ? '#e67e22' : '#e74c3c';
const scoreLabel = (v) => v >= 80 ? '좋음' : v >= 60 ? '보통' : '관리필요';

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  return `${yyyy}.${mm}.${dd} ${hh}:${mi}`;
}

function MiniRing({ score }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  const offset = c - (score / 100) * c;
  const color = scoreColor(score);
  return (
    <div className="mini-ring-wrap">
      <svg width="68" height="68" viewBox="0 0 68 68">
        <circle cx="34" cy="34" r={r} stroke="#eef2f5" strokeWidth="6" fill="none" />
        <circle cx="34" cy="34" r={r}
          stroke={color} strokeWidth="6" fill="none"
          strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={offset}
          transform="rotate(-90 34 34)"
          style={{ transition: 'stroke-dashoffset 0.9s ease-out' }}
        />
      </svg>
      <div className="mini-ring-num" style={{ color }}>{score}</div>
    </div>
  );
}

function TrendChart({ items }) {
  if (!items || items.length < 2) return null;
  const sorted = items.slice().reverse(); // 오래된 → 최신
  const w = 600, h = 160, pad = 32;
  const xs = (i) => pad + (i * (w - pad * 2)) / (sorted.length - 1);
  const ys = (v) => h - pad - ((v / 100) * (h - pad * 2));
  const path = sorted.map((it, i) => `${i === 0 ? 'M' : 'L'} ${xs(i)} ${ys(it.score)}`).join(' ');
  const area = `${path} L ${xs(sorted.length - 1)} ${h - pad} L ${xs(0)} ${h - pad} Z`;
  return (
    <div className="trend-card">
      <div className="trend-title">📈 점수 추이</div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="160" preserveAspectRatio="none">
        <defs>
          <linearGradient id="trend-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2ecc71" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#2ecc71" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0, 50, 100].map((v) => (
          <g key={v}>
            <line x1={pad} y1={ys(v)} x2={w - pad} y2={ys(v)} stroke="#eef2f5" strokeDasharray="4 4" />
            <text x={8} y={ys(v) + 4} fontSize="10" fill="#7f8c8d">{v}</text>
          </g>
        ))}
        <path d={area} fill="url(#trend-grad)" />
        <path d={path} fill="none" stroke="#27ae60" strokeWidth="2.5" strokeLinejoin="round" />
        {sorted.map((it, i) => (
          <circle key={i} cx={xs(i)} cy={ys(it.score)} r="4"
            fill="#fff" stroke={scoreColor(it.score)} strokeWidth="2" />
        ))}
      </svg>
    </div>
  );
}

export default function MyPage() {
  const { currentUser, authFetch } = useAuth();
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await authFetch('/diagnosis/list');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '불러오기 실패');
      setItems(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); /* eslint-disable-next-line */ }, []);

  const openDetail = async (id) => {
    setDetailLoading(true);
    setDetail({ id });
    try {
      const res = await authFetch(`/diagnosis/${id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '상세 조회 실패');
      setDetail(data);
    } catch (e) {
      setDetail(null);
      alert(e.message);
    } finally {
      setDetailLoading(false);
    }
  };

  const removeItem = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm('이 분석 기록을 삭제할까요?')) return;
    try {
      const res = await authFetch(`/diagnosis/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('삭제 실패');
      setItems(prev => prev.filter(it => it.id !== id));
      if (detail?.id === id) setDetail(null);
    } catch (e) {
      alert(e.message);
    }
  };

  const stats = (() => {
    if (!items.length) return null;
    const scores = items.map(i => i.score || 0);
    const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const best = Math.max(...scores);
    const last = items[0]?.score || 0;
    return { avg, best, last, count: items.length };
  })();

  return (
    <div className="mypage">
      <div className="mypage-inner">
        {/* 프로필 */}
        <div className="profile-card">
          <div className="profile-avatar">
            {currentUser?.name?.charAt(0).toUpperCase() || '?'}
          </div>
          <div className="profile-meta">
            <h2>{currentUser?.name || '게스트'}</h2>
            <p>{currentUser?.email}</p>
          </div>
          <button className="profile-cta" onClick={() => navigate('/analysis')}>
            🔍 새 분석 시작
          </button>
        </div>

        {/* 통계 */}
        {stats && (
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">총 분석 횟수</div>
              <div className="stat-value">{stats.count}<span>회</span></div>
            </div>
            <div className="stat-card">
              <div className="stat-label">최근 점수</div>
              <div className="stat-value" style={{ color: scoreColor(stats.last) }}>
                {stats.last}<span>점</span>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">평균 점수</div>
              <div className="stat-value" style={{ color: scoreColor(stats.avg) }}>
                {stats.avg}<span>점</span>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">최고 점수</div>
              <div className="stat-value" style={{ color: scoreColor(stats.best) }}>
                {stats.best}<span>점</span>
              </div>
            </div>
          </div>
        )}

        {items.length >= 2 && <TrendChart items={items} />}

        {/* 기록 리스트 */}
        <div className="records-section">
          <div className="records-header">
            <h3>📁 진단 기록</h3>
            <button className="refresh-btn" onClick={load}>🔄 새로고침</button>
          </div>

          {loading && <div className="records-empty">불러오는 중...</div>}
          {error && <div className="records-error">⚠️ {error}</div>}
          {!loading && !error && items.length === 0 && (
            <div className="records-empty">
              <div className="empty-icon">🪞</div>
              아직 진단 기록이 없어요.<br/>
              <button className="empty-cta" onClick={() => navigate('/analysis')}>
                지금 첫 분석을 시작해 보세요
              </button>
            </div>
          )}

          <div className="record-grid">
            {items.map((it) => (
              <div key={it.id} className="record-card" onClick={() => openDetail(it.id)}>
                <div className="record-thumb">
                  {it.image_url
                    ? <img src={`${API_BASE_URL}${it.image_url}`} alt="진단" />
                    : <div className="record-thumb-empty">🌿</div>}
                </div>
                <div className="record-body">
                  <div className="record-row">
                    <MiniRing score={it.score || 0} />
                    <div className="record-meta">
                      <div className="record-date">{formatDate(it.created_at)}</div>
                      <div className="record-status" style={{ color: scoreColor(it.score || 0) }}>
                        {scoreLabel(it.score || 0)}
                      </div>
                    </div>
                  </div>
                  {it.summary && <p className="record-summary">{it.summary}</p>}
                  <div className="record-actions">
                    <button className="record-detail-btn">자세히 보기 →</button>
                    <button className="record-del-btn" onClick={(e) => removeItem(e, it.id)}>삭제</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 상세 모달 */}
      {detail && (
        <div className="detail-overlay" onClick={() => setDetail(null)}>
          <div className="detail-modal" onClick={(e) => e.stopPropagation()}>
            <div className="detail-header">
              <h3>📊 분석 상세</h3>
              <button className="detail-close" onClick={() => setDetail(null)}>✕</button>
            </div>
            <div className="detail-body">
              {detailLoading || !detail.score ? (
                <div className="detail-loading">불러오는 중...</div>
              ) : (
                <>
                  {detail.image_url && (
                    <img className="detail-image" src={`${API_BASE_URL}${detail.image_url}`} alt="진단 이미지" />
                  )}
                  <div className="detail-date">{formatDate(detail.created_at)}</div>
                  <ScoreVisualization result={detail} />
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
