import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AuthModal from '../components/AuthModal';
import { useState } from 'react';
import './Home.css';

export default function Home() {
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();
  const [showModal, setShowModal] = useState(false);

  const handleNav = (path) => {
    if (!isLoggedIn) { setShowModal(true); return; }
    navigate(path);
  };

  return (
    <div className="home">
      {/* HERO */}
      <section className="hero">
        <div className="hero-badge">✨ AI POWERED SKIN ANALYSIS</div>
        <h1>당신의 피부를 <span>정확하게</span><br/>분석해드립니다</h1>
        <p>사진 한 장으로 피부 타입, 트러블, 수분 상태까지<br/>AI가 전문의 수준으로 진단해드려요.</p>
        <div className="hero-btns">
          <button className="btn-primary" onClick={() => handleNav('/analysis')}>🔍 지금 바로 분석하기</button>
          <button className="btn-outline" onClick={() => handleNav('/chat')}>💬 AI 상담받기</button>
        </div>
      </section>

 {/* HOW IT WORKS */}
      <section className="section how-bg">
        <div className="section-title">
          <div className="label">HOW IT WORKS</div>
          <h2>이용 방법</h2>
          <p>3단계만으로 전문가 수준의 피부 진단을 받아보세요</p>
        </div>
        <div className="steps-row">
          {[
            { n: 1, title: '사진 업로드', desc: '자연광 아래에서 찍은 얼굴 정면 사진을 업로드하세요' },
            { n: 2, title: 'AI 분석', desc: '딥러닝 모델이 12가지 항목을 3초 만에 정밀 분석합니다' },
            { n: 3, title: '결과 확인', desc: '상세한 진단 리포트와 맞춤 케어 방법을 확인하세요' },
            { n: 4, title: 'AI 상담', desc: '궁금한 점은 AI 피부 상담사에게 바로 물어보세요' },
          ].map((s) => (
            <div key={s.n} className="step">
              <div className="step-num">{s.n}</div>
              <h4>{s.title}</h4>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
        <div style={{ textAlign: 'center', marginTop: 40 }}>
          <button className="btn-start" onClick={() => handleNav('/analysis')}>🚀 무료로 시작하기</button>
        </div>
      </section>

      {/* FEATURES */}
      <section className="section">
        <div className="section-title">
          <div className="label">FEATURES</div>
          <h2>SkinAI가 분석하는 것들</h2>
          <p>딥러닝 기반 멀티 진단으로 피부의 모든 것을 분석합니다</p>
        </div>
        <div className="features-grid">
          {[
            { icon: '💧', title: '수분 분석', desc: '피부의 수분 보유량을 분석하여 피부의 수분량을 정확하게 파악합니다.', tag: '', path: '/analysis' },
            { icon: '🔬', title: '트러블 · 모공 진단', desc: '여드름, 블랙헤드, 화이트헤드, 모공 크기 등 트러블 유형과 정도를 정밀 분석합니다.', tag: '', path: '/analysis' },
            { icon: '☀️', title: '색소 · 잡티 분석', desc: '기미, 주근깨, 색소침착 부위를 감지하고 UV 손상 정도를 평가합니다.', tag: '색소침착 지수', path: '/analysis' },
            { icon: '✨', title: '피부 탄력 · 주름', desc: '탄력도와 주름 깊이를 측정하여 피부 나이와 노화 진행 상태를 평가합니다.', tag: '피부 나이 측정', path: '/analysis' },
            { icon: '🎯', title: '민감도 · 알레르기', desc: '피부 민감도를 분석하고 자극에 취약한 부위를 식별하여 관리 방향을 제시합니다.', tag: '민감성 지수', path: '/analysis' },
            { icon: '🤖', title: 'AI 피부 상담', desc: '분석 결과를 바탕으로 AI 피부 전문가와 1:1 맞춤 상담을 진행할 수 있습니다.', tag: '채팅 상담', tagColor: '#e8f4ff', tagTextColor: '#2980b9', path: '/chat' },
          ].map((f, i) => (
            <div key={i} className="feature-card">
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
              <span className="feature-tag" style={f.tagColor ? { background: f.tagColor, color: f.tagTextColor } : {}}>{f.tag}</span>
            </div>
          ))}
        </div>
      </section>

      {/* FOOTER */}
      <footer>
        <div className="footer-inner">
          <div className="footer-brand">
            <h3>🌿 SkinAI</h3>
            <p>AI 기술로 피부 건강을 지키는<br/>스마트 피부 분석 플랫폼입니다.</p>
          </div>
          <div className="footer-col">
            <h4>서비스</h4>
            <span onClick={() => handleNav('/analysis')}>AI 피부 분석</span>
            <span onClick={() => handleNav('/chat')}>AI 피부 상담</span>
          </div>
          <div className="footer-col">
            <h4>안내</h4>
            <span>이용약관</span>
            <span>개인정보처리방침</span>
            <span>고객센터</span>
          </div>
        </div>
        <div className="disclaimer">⚕️ <strong>의료 면책 고지:</strong> SkinAI의 분석 결과는 참고용 정보이며 의학적 진단을 대체하지 않습니다.</div>
        <div className="footer-bottom">
          <span>© 2025 SkinAI. All rights reserved.</span>
          <span>Made with Monkey for healthy skin</span>
        </div>
      </footer>

      {showModal && <AuthModal onClose={() => setShowModal(false)} />}
    </div>
  );
}
