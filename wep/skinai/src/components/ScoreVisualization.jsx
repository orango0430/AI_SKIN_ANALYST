import './ScoreVisualization.css';

const scoreColor = (v) => v >= 80 ? '#27ae60' : v >= 60 ? '#e67e22' : '#e74c3c';
const scoreLabel = (v) => v >= 80 ? '좋음' : v >= 60 ? '보통' : '관리필요';

function ScoreCircle({ score }) {
  const r = 70;
  const c = 2 * Math.PI * r;
  const offset = c - (score / 100) * c;
  const color = scoreColor(score);
  return (
    <div className="score-ring-wrap">
      <svg width="170" height="170" viewBox="0 0 170 170" className="score-ring">
        <defs>
          <linearGradient id="ring-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#2ecc71" />
            <stop offset="100%" stopColor="#1e8449" />
          </linearGradient>
        </defs>
        <circle cx="85" cy="85" r={r} stroke="#eef2f5" strokeWidth="14" fill="none" />
        <circle
          cx="85" cy="85" r={r}
          stroke={score >= 60 ? 'url(#ring-grad)' : color}
          strokeWidth="14"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          transform="rotate(-90 85 85)"
          style={{ transition: 'stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div className="score-ring-text">
        <div className="score-num">{score}<span className="score-unit-inline">점</span></div>
      </div>
    </div>
  );
}

function RadarChart({ details }) {
  if (!details?.length) return null;
  const size = 240, cx = size / 2, cy = size / 2, r = 90;
  const n = details.length;
  const angle = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const point = (i, val) => {
    const rad = (val / 100) * r;
    return [cx + Math.cos(angle(i)) * rad, cy + Math.sin(angle(i)) * rad];
  };
  const polygon = details.map((d, i) => point(i, d.value).join(',')).join(' ');
  const grids = [0.25, 0.5, 0.75, 1].map((s) => {
    const pts = details.map((_, i) => {
      const x = cx + Math.cos(angle(i)) * r * s;
      const y = cy + Math.sin(angle(i)) * r * s;
      return `${x},${y}`;
    }).join(' ');
    return <polygon key={s} points={pts} fill="none" stroke="#e8eef2" strokeWidth="1" />;
  });
  const axes = details.map((_, i) => {
    const [x, y] = [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];
    return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#e8eef2" strokeWidth="1" />;
  });
  const labels = details.map((d, i) => {
    const [x, y] = [cx + Math.cos(angle(i)) * (r + 18), cy + Math.sin(angle(i)) * (r + 18)];
    return (
      <text key={i} x={x} y={y} className="radar-label" textAnchor="middle" dominantBaseline="middle">
        {d.label}
      </text>
    );
  });
  return (
    <div className="radar-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="radar-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#2ecc71" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#1e8449" stopOpacity="0.25" />
          </linearGradient>
        </defs>
        {grids}
        {axes}
        <polygon points={polygon} fill="url(#radar-fill)" stroke="#27ae60" strokeWidth="2" />
        {details.map((d, i) => {
          const [x, y] = point(i, d.value);
          return <circle key={i} cx={x} cy={y} r="4" fill="#fff" stroke="#27ae60" strokeWidth="2" />;
        })}
        {labels}
      </svg>
    </div>
  );
}

export default function ScoreVisualization({ result }) {
  const score = result.score ?? 0;
  const details = result.details || [];
  const issues = result.issues || [];
  return (
    <div className="result-content">
      <div className="skin-score-hero">
        <ScoreCircle score={score} />
        <div className="score-info-block">
          <h3>종합 피부 점수</h3>
          <div className="score-status" style={{ color: scoreColor(score) }}>
            {scoreLabel(score)}
          </div>
          {result.summary && <p className="score-summary">{result.summary}</p>}
        </div>
      </div>

      {details.length > 0 && (
        <>
          <div className="section-divider"><span>레이더 분석</span></div>
          <RadarChart details={details} />
        </>
      )}

      {details.length > 0 && (
        <div className="progress-section">
          <h4>세부 항목 분석</h4>
          {details.map((item) => {
            const c = scoreColor(item.value);
            return (
              <div key={item.label} className="progress-item">
                <div className="label-row">
                  <span>{item.label}</span>
                  <span className="value-row">
                    {item.grade && (
                      <span
                        className="grade-tag"
                        style={{
                          borderColor: c,
                          color: c,
                        }}
                      >
                        {item.grade}
                      </span>
                    )}
                    <span style={{ color: c, fontWeight: 700 }}>{item.value}점</span>
                  </span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${item.value}%`,
                      background: item.value >= 80
                        ? 'linear-gradient(90deg,#2ecc71,#1e8449)'
                        : item.value >= 60
                          ? 'linear-gradient(90deg,#f39c12,#e67e22)'
                          : 'linear-gradient(90deg,#e74c3c,#c0392b)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {issues.length > 0 && (
        <div className="issue-list">
          <div className="issue-title">주요 발견 사항</div>
          {issues.map((issue, i) => (
            <div key={i} className="issue-item">
              <div className="issue-emoji">{issue.emoji}</div>
              <div className="issue-info">
                <h4>{issue.title}</h4>
                <p>{issue.desc}</p>
              </div>
              <span className={`issue-level level-${issue.levelClass}`}>{issue.level}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
